import numpy as np

from sklearn.neighbors import NearestNeighbors
from knn.nearest_neighbors import NearestNeighborsFinder


class KNNClassifier:
    EPS = 1e-5

    def __init__(self, n_neighbors, algorithm='my_own', metric='euclidean', weights='uniform'):
        if algorithm == 'my_own':
            finder = NearestNeighborsFinder(n_neighbors=n_neighbors, metric=metric)
        elif algorithm in ('brute', 'ball_tree', 'kd_tree',):
            finder = NearestNeighbors(n_neighbors=n_neighbors, algorithm=algorithm, metric=metric)
        else:
            raise ValueError("Algorithm is not supported", metric)

        if weights not in ('uniform', 'distance'):
            raise ValueError("Weighted algorithm is not supported", weights)

        self._finder = finder
        self._weights = weights

    def fit(self, X, y=None):
        self._finder.fit(X)
        self._labels = np.asarray(y)
        return self

    def _predict_precomputed(self, indices, distances):
        if self._weights == 'uniform':
            labels_each = self._labels[indices]
            ans = np.apply_along_axis(lambda x: np.bincount(x).argmax(), 1, labels_each)
            return ans
        else:
            inv_dists = np.apply_along_axis(lambda x: 1 / (x + self.EPS), 1, distances)
            ans = []
            labels_each = self._labels[indices]
            for i in range(indices.shape[0]):
                unq, ids = np.unique(labels_each[i], return_inverse=True)
                a = list(zip(unq, np.bincount(ids, inv_dists[i])))
                ans.append(max(a, key=lambda x: x[1])[0])
            return ans

    def kneighbors(self, X, return_distance=False):
        return self._finder.kneighbors(X, return_distance=return_distance)

    def predict(self, X):
        distances, indices = self.kneighbors(X, return_distance=True)
        return self._predict_precomputed(indices, distances)


class BatchedKNNClassifier(KNNClassifier):
    '''
    Нам нужен этот класс, потому что мы хотим поддержку обработки батчами
    в том числе для классов поиска соседей из sklearn
    '''

    def __init__(self, n_neighbors, algorithm='my_own', metric='euclidean', weights='uniform', batch_size=None):
        KNNClassifier.__init__(
            self,
            n_neighbors=n_neighbors,
            algorithm=algorithm,
            weights=weights,
            metric=metric,
        )
        self._batch_size = batch_size

    def kneighbors(self, X, return_distance=False):
        if self._batch_size is None or self._batch_size >= X.shape[0]:
            return super().kneighbors(X, return_distance=return_distance)
        else:
            if return_distance:
                split_idx = np.arange(self._batch_size, X.shape[0], self._batch_size)
                batched_X = np.split(X, split_idx)
                dists_old, inds_old = super().kneighbors(batched_X[0], return_distance)
                for i in range(1, len(batched_X)):
                    dists_new, inds_new = super().kneighbors(batched_X[i], return_distance)
                    inds_old = np.vstack((inds_old, inds_new))
                    dists_old = np.vstack((dists_old, dists_new))
                return dists_old, inds_old
            else:

                split_idx = np.arange(self._batch_size, X.shape[0], self._batch_size)
                batched_X = np.split(X, split_idx)
                inds_old = super().kneighbors(batched_X[0])
                for i in range(1, len(batched_X)):
                    inds_new = super().kneighbors(batched_X[i])
                    inds_old = np.vstack((inds_old, inds_new))
                return inds_old
