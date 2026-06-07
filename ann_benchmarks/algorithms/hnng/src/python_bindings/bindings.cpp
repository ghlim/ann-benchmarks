#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "../hnnglib/hnnglib.h"

namespace py = pybind11;

// Parallel-for borrowed from hnswlib/nmslib. Used only for query parallelism;
// addItems holds a global lock inside the index, so multi-threaded inserts
// would serialize anyway in v1.
template<class Function>
inline void ParallelFor(size_t start, size_t end, size_t numThreads, Function fn) {
    if (numThreads <= 0) numThreads = std::thread::hardware_concurrency();
    if (numThreads == 1) {
        for (size_t id = start; id < end; id++) fn(id, 0);
        return;
    }
    std::vector<std::thread> threads;
    std::atomic<size_t> current(start);
    std::exception_ptr lastException = nullptr;
    std::mutex lastExceptMutex;
    for (size_t threadId = 0; threadId < numThreads; ++threadId) {
        threads.emplace_back([&, threadId] {
            while (true) {
                size_t id = current.fetch_add(1);
                if (id >= end) break;
                try { fn(id, threadId); }
                catch (...) {
                    std::unique_lock<std::mutex> l(lastExceptMutex);
                    lastException = std::current_exception();
                    current = end;
                    break;
                }
            }
        });
    }
    for (auto& t : threads) t.join();
    if (lastException) std::rethrow_exception(lastException);
}

template<typename dist_t>
class Index {
 public:
    Index(const std::string& space_name, int dim) : space_name_(space_name), dim_(dim) {
        normalize_ = false;
        if (space_name == "l2") {
            space_ = new hnnglib::L2Space(dim);
        } else if (space_name == "ip") {
            space_ = new hnnglib::InnerProductSpace(dim);
        } else if (space_name == "cosine") {
            space_ = new hnnglib::InnerProductSpace(dim);
            normalize_ = true;
        } else {
            throw std::runtime_error("Unknown space: " + space_name);
        }
        num_threads_default_ = std::thread::hardware_concurrency();
    }

    ~Index() {
        delete space_;
        delete alg_;
    }

    void init_index(size_t max_elements) {
        if (alg_) throw std::runtime_error("Index already initialized");
        alg_ = new hnnglib::HierarchicalNNG<dist_t>(space_, max_elements);
        cur_l_ = 0;
    }

    void load_index(const std::string& path, size_t /*max_elements*/ = 0) {
        delete alg_;
        alg_ = new hnnglib::HierarchicalNNG<dist_t>(space_, path);
        cur_l_ = alg_->getCurrentElementCount();
    }

    void save_index(const std::string& path) {
        ensureInit_();
        alg_->saveIndex(path);
    }

    void set_num_threads(int n) { num_threads_default_ = n; }

    size_t get_max_elements()    const { return alg_ ? alg_->getMaxElements() : 0; }
    size_t get_current_count()   const { return alg_ ? alg_->getCurrentElementCount() : 0; }

    void normalize_vector_(const float* in, float* out) {
        float norm = 0.f;
        for (int i = 0; i < dim_; ++i) norm += in[i] * in[i];
        norm = 1.0f / (std::sqrt(norm) + 1e-30f);
        for (int i = 0; i < dim_; ++i) out[i] = in[i] * norm;
    }

    void add_items(py::object input, py::object ids_obj = py::none(),
                   int /*num_threads*/ = -1) {
        ensureInit_();
        py::array_t<dist_t, py::array::c_style | py::array::forcecast> arr(input);
        auto buf = arr.request();
        size_t rows, features;
        if (buf.ndim == 2)      { rows = buf.shape[0]; features = buf.shape[1]; }
        else if (buf.ndim == 1) { rows = 1;            features = buf.shape[0]; }
        else throw std::runtime_error("data must be a 1d/2d array");
        if (static_cast<int>(features) != dim_)
            throw std::runtime_error("wrong dimensionality of the vectors");

        std::vector<size_t> ids;
        if (!ids_obj.is_none()) {
            py::array_t<size_t, py::array::c_style | py::array::forcecast> ia(ids_obj);
            auto ib = ia.request();
            if (ib.ndim == 1 && static_cast<size_t>(ib.shape[0]) == rows) {
                ids.resize(rows);
                for (size_t i = 0; i < rows; ++i) ids[i] = ia.data()[i];
            } else if (ib.ndim == 0 && rows == 1) {
                ids.push_back(*ia.data());
            } else {
                throw std::runtime_error("wrong dimensionality of the labels");
            }
        }

        // Insertion is serialized inside the index (global_lock_). We add one
        // row at a time on the calling thread to stay GIL-friendly.
        std::vector<float> norm_buf(normalize_ ? dim_ : 0);
        for (size_t row = 0; row < rows; ++row) {
            size_t id = ids.size() ? ids[row] : (cur_l_ + row);
            float* vec = (float*)arr.data(row);
            if (normalize_) {
                normalize_vector_(vec, norm_buf.data());
                vec = norm_buf.data();
            }
            alg_->addPoint(vec, static_cast<hnnglib::labeltype>(id));
        }
        cur_l_ += rows;
    }

    py::object knn_query(py::object input, size_t k = 1, int num_threads = -1) {
        ensureInit_();
        py::array_t<dist_t, py::array::c_style | py::array::forcecast> arr(input);
        auto buf = arr.request();
        size_t rows, features;
        if (buf.ndim == 2)      { rows = buf.shape[0]; features = buf.shape[1]; }
        else if (buf.ndim == 1) { rows = 1;            features = buf.shape[0]; }
        else throw std::runtime_error("data must be a 1d/2d array");
        if (static_cast<int>(features) != dim_)
            throw std::runtime_error("wrong dimensionality of the vectors");

        if (num_threads <= 0) num_threads = num_threads_default_;
        if (rows <= static_cast<size_t>(num_threads) * 4) num_threads = 1;

        auto* labels = new hnnglib::labeltype[rows * k];
        auto* dists  = new dist_t[rows * k];

        {
            py::gil_scoped_release nogil;
            std::vector<float> norm_buf(normalize_ ? num_threads * dim_ : 0);
            ParallelFor(0, rows, num_threads, [&](size_t row, size_t tid) {
                const float* query = (const float*)arr.data(row);
                if (normalize_) {
                    float* nb = norm_buf.data() + tid * dim_;
                    normalize_vector_(query, nb);
                    query = nb;
                }
                auto result = alg_->searchKnn(const_cast<void*>(static_cast<const void*>(query)), k);
                if (result.size() != k) {
                    // Fewer than k stored — pad with sentinels.
                    while (result.size() < k)
                        result.push({std::numeric_limits<dist_t>::max(), 0});
                }
                for (int i = static_cast<int>(k) - 1; i >= 0; --i) {
                    labels[row * k + i] = result.top().second;
                    dists [row * k + i] = result.top().first;
                    result.pop();
                }
            });
        }

        py::capsule free_l(labels, [](void* p) { delete[] (hnnglib::labeltype*)p; });
        py::capsule free_d(dists,  [](void* p) { delete[] (dist_t*)p; });

        return py::make_tuple(
            py::array_t<hnnglib::labeltype>(
                {rows, k},
                {k * sizeof(hnnglib::labeltype), sizeof(hnnglib::labeltype)},
                labels, free_l),
            py::array_t<dist_t>(
                {rows, k},
                {k * sizeof(dist_t), sizeof(dist_t)},
                dists, free_d));
    }

 private:
    void ensureInit_() const {
        if (!alg_) throw std::runtime_error("Index not initialized (call init_index first)");
    }

    std::string space_name_;
    int  dim_;
    bool normalize_;
    int  num_threads_default_;
    size_t cur_l_ = 0;
    hnnglib::HierarchicalNNG<dist_t>* alg_ = nullptr;
    hnnglib::SpaceInterface<float>*   space_ = nullptr;
};

PYBIND11_MODULE(hnnglib, m) {
    m.doc() = "Hierarchical Nearest Neighbor Graph (HNNG) — Python bindings";

    py::class_<Index<float>>(m, "Index")
        .def(py::init<const std::string&, int>(), py::arg("space"), py::arg("dim"))
        .def("init_index",      &Index<float>::init_index,        py::arg("max_elements"))
        .def("add_items",       &Index<float>::add_items,         py::arg("data"),
             py::arg("ids") = py::none(), py::arg("num_threads") = -1)
        .def("knn_query",       &Index<float>::knn_query,         py::arg("data"),
             py::arg("k") = 1, py::arg("num_threads") = -1)
        .def("save_index",      &Index<float>::save_index,        py::arg("path"))
        .def("load_index",      &Index<float>::load_index,        py::arg("path"),
             py::arg("max_elements") = 0)
        .def("set_num_threads", &Index<float>::set_num_threads,   py::arg("num_threads"))
        .def("get_max_elements",  &Index<float>::get_max_elements)
        .def("get_current_count", &Index<float>::get_current_count)
        .def("__repr__", [](const Index<float>&) { return "<hnnglib.Index>"; });
}
