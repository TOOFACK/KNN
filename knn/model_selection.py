from collections import defaultdict

import numpy as np

from sklearn.model_selection import KFold, BaseCrossValidator
from sklearn.metrics import accuracy_score

from knn.classification import BatchedKNNClassifier


def knn_cross_val_score(X, y, k_list, scoring, cv=None, **kwargs):
    y = np.asarray(y)

    if scoring == "accuracy":
        scorer = accuracy_score
    else:
        raise ValueError("Unknown scoring metric", scoring)

    if cv is None:
        cv = KFold(n_splits=5)
    elif not isinstance(cv, BaseCrossValidator):
        raise TypeError("cv should be BaseCrossValidator instance", type(cv))


    scores = {}
    for i in k_list:
        scores[i] = []

    for train, test in cv.split(X):
        X_train = X[train]
        X_test = X[test]

        y_train = y[train]
        y_true = y[test]

        clf = BatchedKNNClassifier(max(k_list), **kwargs)

        clf.fit(X_train, y_train)

        dist, inds = clf.kneighbors(X_test, return_distance=True)

        for k in k_list:
            anses = clf._predict_precomputed(inds[:, :k], dist[:, :k])

            scores[k].append(scorer(y_true, anses))
    for k in k_list:
        scores[k] = np.array(scores[k])

    return scores
