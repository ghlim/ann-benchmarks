"""ann-benchmarks algorithm module for hNNG (exact hierarchical NNG search).

Drop-in for ``ann_benchmarks/algorithms/hnng/module.py``.

hNNG is an *exact* k-NN method (recall = 1.0). It therefore produces a single
(recall ~= 1.0, QPS) point per dataset -- there is no M/ef recall sweep like an
approximate method (HNSW etc.) would have. It is directly comparable to the
``bf`` / ``bruteforce-blas`` baselines.

Metric mapping: ``euclidean`` -> internal "l2". ``angular`` is *not* a metric,
but L2 distance on L2-normalized vectors induces the same ranking and *is* a
metric, so we normalize at fit/query time and run L2 internally (keeps the
triangle-inequality pruning exact). See IMPLEMENTATION_PLAN.md sec 1.4.

Backend selection: prefer the compiled C++/pybind11 core (``import hnng``) and
fall back to the pure-Python reference (``hnng_ref``) if it is not installed.
The ``Index`` API (``Index(metric).build(X)`` + ``knn_query(q, k) -> indices``)
is identical, so this module is a drop-in either way.
"""

import numpy as np

# Unified angular zero-vector rule (shared with hnng_ref/metrics.py and
# cpp/include/hnng/distance.hpp): L2-normalize by dividing by max(||x||, _EPS).
# A true zero vector therefore maps to the zero vector. Angular distance is
# UNDEFINED for zero vectors (they have no direction); this eps-guard is a
# defined, documented fallback that avoids a divide-by-zero and keeps all three
# code paths byte-for-byte consistent -- it does NOT assign a meaningful
# direction. The same constant is used everywhere.
_ANGULAR_EPS = 1e-30

# --- hNNG backend ---------------------------------------------------------
# Prefer the compiled C++ core; fall back to the pure-Python reference.
try:
    import hnng as hnng_backend          # compiled C++/pybind11 core (fast)
    HNNG_BACKEND = "hnng (C++)"
except ImportError:  # pragma: no cover - reference fallback
    import hnng_ref as hnng_backend      # pure-Python reference (correct, slow)
    HNNG_BACKEND = "hnng_ref (Python)"

# BaseANN lives at ann_benchmarks/algorithms/base/module.py. Guard the relative
# import so this file is ALSO runnable standalone for a quick smoke check
# (python module.py) outside an ann-benchmarks checkout.
try:
    from ..base.module import BaseANN
except (ImportError, ValueError):  # pragma: no cover - standalone fallback
    class BaseANN:  # minimal stand-in mirroring the real interface
        def done(self):
            pass

        def get_memory_usage(self):
            return None

        def batch_query(self, X, n):
            self.res = [self.query(q, n) for q in X]

        def get_batch_results(self):
            return self.res

        def __str__(self):
            return getattr(self, "name", self.__class__.__name__)


class HNNG(BaseANN):
    """Exact hierarchical Nearest Neighbor Graph for ann-benchmarks."""

    def __init__(self, metric, params=None):
        # ann-benchmarks passes the dataset metric string ("euclidean"/"angular")
        # plus a per-run-group params dict (the run_groups `args`). hNNG is exact
        # and has no recall-tuning knobs; the one knob we expose is the BUILD MODE
        # ("batch" = Index.build(X) bottom-up clustering; "incremental" = n
        # sequential Index.insert(v) calls = the true AIhNNGc incremental path).
        # Both yield recall 1.0; they differ only in build time, so this lets the
        # harness compare batch vs incremental construction on the same dataset.
        self._metric = {"angular": "angular", "euclidean": "l2"}[metric]
        self._ann_metric = metric
        self._params = params or {}
        self._build_mode = str(self._params.get("build", "batch"))
        self._index = None
        self.name = "hnng" + ("-incr" if self._build_mode == "incremental" else "")

    def fit(self, X):
        X = np.ascontiguousarray(X, dtype=np.float32)
        if self._metric == "angular":
            # L2-normalize -> L2 distance reproduces cosine ranking (and is a
            # metric). Divide by max(||x||, _ANGULAR_EPS): the unified zero-vector
            # rule (see top of file); a zero row maps to the zero vector.
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms = np.maximum(norms, _ANGULAR_EPS)
            X = (X / norms).astype(np.float32)
        self._index = hnng_backend.Index(self._metric)
        if self._build_mode == "incremental":
            # True incremental construction: one insert() per point (AIhNNGc).
            for v in X:
                self._index.insert(np.ascontiguousarray(v, dtype=np.float32))
        else:
            self._index.build(X)

    def query(self, q, n):
        q = np.ascontiguousarray(q, dtype=np.float32)
        if self._metric == "angular":
            # Unified zero-vector rule: divide by max(||q||, _ANGULAR_EPS).
            nrm = max(float(np.linalg.norm(q)), _ANGULAR_EPS)
            q = q / nrm
        # knn_query on a single (d,) vector returns a (k,) array of INDICES.
        return self._index.knn_query(q, n)

    def batch_query(self, X, n):
        # Run the whole (m, d) batch through the C++ core in ONE call: knn_query
        # parallelizes independent queries across all hardware threads internally
        # (num_threads=0 -> auto), with the GIL released. Recall is identical to
        # the sequential path (each query is deterministic). Use ann-benchmarks
        # --batch to exercise this.
        X = np.ascontiguousarray(X, dtype=np.float32)
        if self._metric == "angular":
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms = np.maximum(norms, _ANGULAR_EPS)
            X = (X / norms).astype(np.float32)
        self._batch_res = self._index.knn_query(X, n)

    def get_batch_results(self):
        return self._batch_res

    def get_memory_usage(self):
        """Index memory footprint in KILOBYTES (the ann-benchmarks unit).

        Prefer the compiled core's exact structural accounting
        (``memory_bytes()`` -- coordinate store + per-level element/cluster
        buffers); fall back to the BaseANN default (process RSS) when the
        pure-Python reference backend is in use (it exposes no such accessor).
        """
        idx = self._index
        if idx is not None and hasattr(idx, "memory_bytes"):
            return idx.memory_bytes() / 1024.0
        return super().get_memory_usage()

    def __str__(self):
        return "hnng(%s,%s)" % (self._ann_metric, self._build_mode)


# --------------------------------------------------------------------------
# Standalone smoke check: `python module.py` builds a tiny index and verifies
# the result matches a brute-force oracle (recall 1.0). Not used by
# ann-benchmarks; purely to confirm the file imports and runs.
if __name__ == "__main__":
    rng = np.random.RandomState(0)
    n, d, k = 300, 8, 10
    X = rng.randn(n, d).astype(np.float32)
    for metric in ("euclidean", "angular"):
        algo = HNNG(metric)
        algo.fit(X)
        # Brute-force oracle on the same prepared data.
        Xp = X.copy()
        if metric == "angular":
            Xp = Xp / np.linalg.norm(Xp, axis=1, keepdims=True)
        mismatches = 0
        for i in rng.choice(n, size=25, replace=False):
            q = X[i]
            got = set(int(j) for j in algo.query(q, k))
            qp = Xp[i]
            dists = np.linalg.norm(Xp - qp, axis=1)
            exp = set(int(j) for j in np.argsort(dists, kind="stable")[:k])
            mismatches += got != exp
        status = "PASS" if mismatches == 0 else "FAIL"
        print(f"[{metric}] mismatches={mismatches}/25 -> {status}")
