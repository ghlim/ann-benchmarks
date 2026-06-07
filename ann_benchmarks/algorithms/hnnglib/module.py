import hnnglib
import numpy as np

from ..base.module import BaseANN


class HnngLib(BaseANN):
    """ann-benchmarks wrapper for hnnglib (top-level d:/hnng/hnnglib).

    Hierarchical Nearest Neighbor Graph — exact k-NN via cluster-radius
    pruning over a hierarchy of 1-NNGs. Has no query-time tuning knob, so a
    single recall/QPS point is produced per dataset.
    """

    def __init__(self, metric, method_param=None):
        self.metric = {"angular": "cosine", "euclidean": "l2"}[metric]
        self.name = "hnnglib"

    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)
        self.p = hnnglib.Index(space=self.metric, dim=X.shape[1])
        self.p.init_index(max_elements=len(X))
        self.p.add_items(X, np.arange(len(X)))

    def query(self, v, n):
        v = np.asarray(v, dtype=np.float32)
        labels = self.p.knn_query(np.expand_dims(v, axis=0), k=n)[0]
        return labels
