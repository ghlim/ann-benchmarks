import hnnglib
import numpy as np

from ..base.module import BaseANN


class Hnng(BaseANN):
    """ann-benchmarks wrapper for hnng (branch Code20260528, d:/hnng/hnng).

    Research-grade HNNG with SIMD distances, incremental hierarchy
    maintenance and invariant checks. Exposes the same hnswlib-style Python
    API (module name ``hnnglib``). No query-time tuning knob, so a single
    recall/QPS point is produced per dataset.
    """

    def __init__(self, metric, method_param=None):
        self.metric = {"angular": "cosine", "euclidean": "l2"}[metric]
        self.name = "hnng"

    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)
        self.p = hnnglib.Index(space=self.metric, dim=X.shape[1])
        self.p.init_index(max_elements=len(X))
        self.p.add_items(X, np.arange(len(X)))
        if hasattr(self.p, "set_num_threads"):
            self.p.set_num_threads(1)

    def query(self, v, n):
        v = np.asarray(v, dtype=np.float32)
        labels = self.p.knn_query(np.expand_dims(v, axis=0), k=n)[0]
        return np.asarray(labels).reshape(-1)
