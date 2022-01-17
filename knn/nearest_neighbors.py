import numpy as np

from knn.distances import euclidean_distance, cosine_distance


def get_best_ranks(ranks, top, axis=1, return_ranks=False):
    raise NotImplementedError()


class NearestNeighborsFinder:
    def __init__(self, n_neighbors, metric="euclidean"):
        self.n_neighbors = n_neighbors

        if metric == "euclidean":
            self._metric_func = euclidean_distance
        elif metric == "cosine":
            self._metric_func = cosine_distance
        else:
            raise ValueError("Metric is not supported", metric)
        self.metric = metric

    def fit(self, X, y=None):
        self._X = X
        return self

    def kneighbors(self, X, return_distance=False):
        m_distances = self._metric_func(X, self._X)
        k = self.n_neighbors
        inds = np.argpartition(m_distances, k-1, axis=1)[:, :k]
        dists = np.take_along_axis(m_distances, inds, axis=1)
        dists_ord = np.argsort(dists, axis=1)[:, :k]
        ord_inds = np.take_along_axis(inds, dists_ord, axis=1)
        dists = np.take_along_axis(m_distances, ord_inds, axis=1)

        if return_distance:
            return dists, ord_inds
        else:
            return ord_inds
