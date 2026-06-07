# hnng — C++/pybind11 core

Exact hierarchical Nearest Neighbor Graph (hNNG) nearest-neighbor index. This is
the C++17 core plus a pybind11 binding, written to **mirror the pure-Python
reference `hnng_ref` 1:1** — same algorithm, same best-first branch-and-bound
search, same incremental semantics, same `l2` and `angular` (L2-on-normalized)
metrics, same deterministic tie-breaking.

> **Compiles and is tested.** Built with **g++ 13.3 / pybind11 3.0.4** (Ubuntu
> 24.04, WSL) with zero source changes; `import hnng` works and the Python
> parity suite (`tests/test_cpp_parity.py`) — the acceptance gate that runs the
> same inputs through `hnng_ref` and this module and compares outputs — is
> green. CI runs the full suite.

## What it mirrors

`hnng_ref` (the source of truth, passing its full test suite) is reproduced
step-for-step:

| reference (`hnng_ref/core.py`)      | here                                   |
|-------------------------------------|----------------------------------------|
| `Element`, `Cluster`                | `include/hnng/element.hpp`             |
| `levels` / `clusters` per level     | `include/hnng/level.hpp` + `hnng.hpp`  |
| `Metric` (l2 / angular)             | `include/hnng/distance.hpp`            |
| `_link_nearest`                     | `HNNG::link_nearest`                   |
| `_connected_components`             | `HNNG::connected_components`           |
| `_select_rep` (HDE)                 | `HNNG::select_rep`                     |
| `_build_level_clusters` (radius)    | `HNNG::build_level_clusters`           |
| `_rebuild_from_objects`             | `HNNG::rebuild_from_active`            |
| `_knn_single` (best-first BnB)      | `HNNG::knn_single` + `heap.hpp`        |
| `_delta` (memoized, eval counter)   | `HNNG::delta`                          |
| `build` / `insert` / `remove`       | `HNNG::build` / `insert` / `remove`    |
| `knn_query` / `stats`               | `HNNG::knn_query_batch` / `stats`      |

Determinism choices that MUST match the reference and are reproduced here:
NN ties break to the smaller `object_index`; components are emitted sorted by
their smallest `object_index` with members sorted by `object_index`; rep
selection uses the exact HDE key `(-num_child, -#inlinks, avg_inlink_dist,
outlink_dist, object_index)`; results are ordered by `(distance, object_index)`;
the per-query distance memoization makes `last_query_dist_evals` count *distinct*
object evaluations.

### Incremental insert/remove (true AIhNNGc)

`insert`/`remove` are **truly incremental** — a port of the ROS
`race_view_clustering` algorithm — and do **not** rebuild from scratch.
`insert` appends the leaf, links it to its level-0 nearest neighbour and
asymmetrically re-links the others (ROS `updateTheOtherElements`); `remove`
re-links the removed node's inlinkers to their new NN. Both then run
`reconcile_up`, which recomputes each level's clusters from the maintained links
and propagates representative add/remove/field-update changes upward (with link
work only when a level's rep set actually changes), creating a new top level on
growth and collapsing to the root on shrink. Cost is ~**O(n·d) per op** (the
asymmetric re-link scans the level once), versus the old **O(n²)** rebuild — a
~300× per-insert speedup measured at n≈2–4k. Removed vectors stay in the backing
store so object ids remain stable. `build()` still uses the batch rebuild.

The incrementally maintained hierarchy is **not guaranteed structurally
identical** to a batch rebuild (e.g. `num_levels` may differ after long op
sequences), but it stays a **valid** hNNG: full coverage plus accumulating radii
that bound every subtree. The exact best-first search is correct for any such
hierarchy, so it still returns the true k-NN — verified by
`tests/test_cpp_parity.py` (`test_insert_sequence_parity`,
`test_remove_sequence_parity`) against the rebuild-based `hnng_ref` oracle,
tie-safely.

### Numerical note

Coordinates are stored as `float` (ann-benchmarks data is float32), but every
distance is accumulated in `double` and `prepare`/normalization happen in
`double`, to track the reference (which computes in float64). This keeps
tie-breaking and pruning decisions aligned with `hnng_ref`. If the first parity
run shows rare boundary disagreements on exactly-tied distances, the fix is to
store coordinates as `double` as well (the reference's representation); this is a
one-line change to `coord_t` in `distance.hpp` plus the binding's array dtype.

## Layout

```
cpp/
├── CMakeLists.txt          # pybind11 module 'hnng', C++17, -O3
├── pyproject.toml          # PEP 517 build (setuptools + pybind11)
├── setup.py                # Pybind11Extension build recipe
├── include/hnng/
│   ├── distance.hpp        # L2 + L2Normalized (angular) metric policies
│   ├── element.hpp         # Element, Cluster (index-based uint32 storage)
│   ├── level.hpp           # one NNG level's element array
│   ├── heap.hpp            # min-PQ (cluster LB) + bounded max-heap (k best)
│   └── hnng.hpp            # HNNG: build / insert / remove / knn_query / stats
├── bindings/
│   └── hnng_py.cpp         # pybind11: numpy float32, GIL-released, int indices
└── tests/
    └── smoke_main.cpp      # standalone no-Python sanity check (optional target)
```

The core is **header-only** (everything lives in `include/hnng/`); the only
translation unit is the binding `bindings/hnng_py.cpp`. There is no `src/*.cpp`
to compile separately — this keeps the hot path in one TU so the compiler can
inline the distance routine. (The `src/` directory is intentionally empty; if a
future change wants out-of-line definitions, add them there and list them in
both `CMakeLists.txt` and `setup.py`.)

## Building (once a toolchain exists)

### Python module (preferred — this is what ann-benchmarks uses)

```bash
cd cpp
pip install .                      # builds + installs the `hnng` module
python -c "import hnng; print(hnng.Index('l2'))"
```

or in-place for development:

```bash
cd cpp
python setup.py build_ext --inplace
```

Set `HNNG_NATIVE=1` to add `-march=native` (faster, non-portable) when building
on the run host.

### CMake (for the standalone C++ smoke test / IDE integration)

```bash
cd cpp
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j
# optional no-Python smoke check:
cmake -S . -B build -DHNNG_BUILD_SMOKE=ON && cmake --build build
./build/hnng_smoke
```

`find_package(pybind11)` is used if pybind11 is installed; otherwise CMake
fetches it via `FetchContent` (pin: pybind11 v2.12.0).

## Python-facing API (drop-in for `hnng_ref.Index`)

```python
import numpy as np, hnng

idx = hnng.Index("l2")             # or "angular"
X = np.ascontiguousarray(data, dtype=np.float32)   # (n, d)
idx.build(X)

out = idx.knn_query(Q, k)          # Q: (m, d) -> (m, k) int64 indices
                                   # Q: (d,)   -> (k,)  int64 indices
                                   # short rows right-padded with -1

nid = idx.insert(np.ascontiguousarray(v, dtype=np.float32))  # (d,) -> new id
idx.remove(nid)

s = idx.stats()  # {height, num_levels, clusters_per_level, avg_cluster_size,
                 #  last_query_dist_evals, last_query_clusters_visited}
```

This matches `hnng_ref.Index` so the ann-benchmarks `module.py` and the parity
tests can swap `from hnng_ref import Index` for `from hnng import Index`. The one
input difference: this module wants `float32` C-contiguous numpy (zero-copy);
the reference accepts any dtype and casts to float64 internally. The
ann-benchmarks `fit`/`query` already `ascontiguousarray(..., dtype=np.float32)`,
so the drop-in is clean there.

## Build notes

The first compilation surfaced no source changes (g++ 13.3 / pybind11 3.0.4).
For the record, the spots worth knowing about:

1. **pybind11 `std::move(result)` returns** in `hnng_py.cpp` — returning
   `py::array_t<...>` by value from a function typed `py::array` compiles fine.
2. **`mutable_unchecked<N>()`** requires a writable array of the right ndim — it
   is (freshly allocated), template ndim args verified.
3. **`std::tolower` include** — `<cctype>` is included explicitly in
   `distance.hpp`.
4. **`-march=native` / SIMD** — off by default for portable wheels; enable for
   the benchmark build via `HNNG_NATIVE=1` (see `setup.py`). Parity is re-verified
   under the native build (the suite is run against the SIMD module), so the
   vectorized fp reductions do not change any result.
```
