// hnng_py.cpp -- pybind11 bindings exposing the hNNG core as `import hnng`.
//
// Compiles and is tested: built with g++ 13.3 / pybind11 3.0.4; the full suite
// (incl. C++/Python parity) runs in CI.
//
// Structural reference: the hnswlib ann-benchmarks binding (bindings.cpp) ->
// accept C-contiguous float32 numpy with zero-copy via request(), RELEASE the
// GIL around the heavy build/query work, and return neighbour INDICES as a
// numpy int array (the BaseANN.query contract).
//
// Public Python API is kept IDENTICAL to hnng_ref.Index so this module is a
// drop-in replacement:
//
//   import hnng
//   idx = hnng.Index("l2")            # or "angular"
//   idx.build(X)                       # X: (n, d) float32, C-contiguous
//   nid = idx.insert(v)                # v: (d,) float32 -> new object id
//   idx.remove(object_id)
//   out = idx.knn_query(Q, k)          # Q: (m, d) or (d,) -> (m, k) or (k,) int64
//   s = idx.stats()                    # dict, same keys as the reference

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "hnng/hnng.hpp"

namespace py = pybind11;

namespace {

using FloatArray =
    py::array_t<float, py::array::c_style | py::array::forcecast>;

// Build from a 2-D (n, d) float32 array.
void index_build(hnng::HNNG& self, FloatArray X) {
    py::buffer_info b = X.request();
    if (b.ndim != 2) {
        throw std::invalid_argument(
            "build expects a 2-D (n, d) array, got ndim=" + std::to_string(b.ndim));
    }
    const auto n = static_cast<std::size_t>(b.shape[0]);
    const auto d = static_cast<std::size_t>(b.shape[1]);
    const float* ptr = static_cast<const float*>(b.ptr);
    {
        py::gil_scoped_release nogil;  // heavy build runs without the GIL
        self.build(ptr, n, d);
    }
}

// Insert a single (d,) float32 vector, return the new object id (int).
std::uint32_t index_insert(hnng::HNNG& self, FloatArray vec) {
    py::buffer_info b = vec.request();
    if (b.ndim != 1) {
        throw std::invalid_argument(
            "insert expects a 1-D (d,) vector, got ndim=" + std::to_string(b.ndim));
    }
    const auto d = static_cast<std::size_t>(b.shape[0]);
    const float* ptr = static_cast<const float*>(b.ptr);
    std::uint32_t new_id;
    {
        py::gil_scoped_release nogil;
        new_id = self.insert(ptr, d);
    }
    return new_id;
}

void index_remove(hnng::HNNG& self, std::uint32_t object_id) {
    py::gil_scoped_release nogil;
    self.remove(object_id);
}

// Exact k-NN. Accepts (m, d) batch or (d,) single. Returns (m, k) int64 indices
// for a batch, or (k,) for a single (d,) query -- matching hnng_ref.knn_query.
// num_threads: 0 = auto (use all hardware threads for a multi-query batch), 1 =
// sequential, N = use N threads. Queries are independent so results are
// identical regardless of thread count.
py::array index_knn_query(hnng::HNNG& self, FloatArray Q, std::size_t k,
                          int num_threads) {
    if (k == 0) throw std::invalid_argument("k must be positive");
    py::buffer_info b = Q.request();

    bool single = (b.ndim == 1);
    std::size_t m;
    std::size_t d;
    if (single) {
        m = 1;
        d = static_cast<std::size_t>(b.shape[0]);
    } else if (b.ndim == 2) {
        m = static_cast<std::size_t>(b.shape[0]);
        d = static_cast<std::size_t>(b.shape[1]);
    } else {
        throw std::invalid_argument(
            "knn_query expects (m, d) or (d,), got ndim=" + std::to_string(b.ndim));
    }

    const float* qptr = static_cast<const float*>(b.ptr);

    // Allocate the output buffer (m * k int64) before releasing the GIL; fill it
    // in the core while the GIL is released, then wrap it as a numpy array.
    std::vector<std::int64_t> out(m * k);
    // Resolve thread count: 0 = auto (all hardware threads for a real batch).
    unsigned nthreads;
    if (num_threads > 0) {
        nthreads = static_cast<unsigned>(num_threads);
    } else {
        unsigned hw = std::thread::hardware_concurrency();
        if (hw == 0) hw = 1;
        nthreads = (m > 1) ? hw : 1;
    }
    {
        py::gil_scoped_release nogil;
        self.knn_query_batch_parallel(qptr, m, d, k, out.data(), nthreads);
    }

    if (single) {
        // Return shape (k,). Brace-list shape avoids ambiguity with other
        // py::array_t constructors.
        py::array_t<std::int64_t> result(
            std::vector<py::ssize_t>{static_cast<py::ssize_t>(k)});
        auto r = result.mutable_unchecked<1>();
        for (std::size_t j = 0; j < k; ++j) r(static_cast<py::ssize_t>(j)) = out[j];
        return std::move(result);
    }
    // Return shape (m, k).
    py::array_t<std::int64_t> result(
        std::vector<py::ssize_t>{static_cast<py::ssize_t>(m),
                                 static_cast<py::ssize_t>(k)});
    auto r = result.mutable_unchecked<2>();
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < k; ++j) {
            r(static_cast<py::ssize_t>(i), static_cast<py::ssize_t>(j)) = out[i * k + j];
        }
    }
    return std::move(result);
}

py::dict index_stats(const hnng::HNNG& self) {
    hnng::Stats s = self.stats();
    py::dict d;
    d["height"] = s.height;
    d["num_levels"] = s.num_levels;
    d["clusters_per_level"] = s.clusters_per_level;  // list[int]
    d["avg_cluster_size"] = s.avg_cluster_size;
    d["last_query_dist_evals"] = s.last_query_dist_evals;
    d["last_query_clusters_visited"] = s.last_query_clusters_visited;
    return d;
}

}  // namespace

PYBIND11_MODULE(hnng, m) {
    m.doc() =
        "Exact hierarchical Nearest Neighbor Graph (hNNG) index. C++ core "
        "mirroring the hnng_ref pure-Python reference 1:1.";

    py::class_<hnng::HNNG>(m, "Index")
        .def(py::init<std::string, std::string>(), py::arg("metric") = "l2",
             py::arg("hde_rule") = "v2026",
             "Create an index. metric is 'l2' or 'angular' (angular = L2 on "
             "L2-normalized vectors, same ranking as cosine). hde_rule selects "
             "the HDE rep rule: 'v2026' (default; num_child first) or 'v2015' "
             "(original ROS; #inlinks first, no num_child term).")
        .def("build", &index_build, py::arg("X"),
             "Build the index from a (n, d) float32 C-contiguous array.")
        .def("insert", &index_insert, py::arg("vec"),
             "Insert a single (d,) vector; returns its new object id.")
        .def("remove", &index_remove, py::arg("object_id"),
             "Remove an object by id from the active hierarchy.")
        .def("knn_query", &index_knn_query, py::arg("Q"), py::arg("k"),
             py::arg("num_threads") = 0,
             "Exact k-NN. Q is (m, d) or (d,); returns int64 indices (m, k) or "
             "(k,), right-padded with -1 when fewer than k objects exist. "
             "num_threads: 0=auto (parallelize a multi-query batch over all "
             "hardware threads), 1=sequential, N=use N threads. Results are "
             "identical regardless of thread count.")
        .def("stats", &index_stats,
             "Return a dict of hierarchy/last-query statistics.")
        .def("memory_bytes",
             [](const hnng::HNNG& self) {
                 py::gil_scoped_release nogil;
                 return self.memory_bytes();
             },
             "Approximate resident byte-size of the index (coordinate store + "
             "per-level element/cluster buffers).")
        .def_property_readonly(
            "metric",
            [](const hnng::HNNG& self) { return std::string(hnng::metric_to_string(self.metric())); })
        .def_property_readonly(
            "hde_rule",
            [](const hnng::HNNG& self) { return std::string(hnng::hde_rule_to_string(self.hde_rule())); });
}
