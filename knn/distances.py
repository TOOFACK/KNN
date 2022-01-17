import numpy as np


def euclidean_distance(x, y):
    m = x.shape[0]
    n = y.shape[0]

    a = (x * x).sum(axis=1)

    a = a.reshape(m, -1)
    a = np.tile(a, (1, n))

    b = (y * y).sum(axis=1)
    b = np.tile(b, (m, 1))

    return np.sqrt(a + b - 2 * x @ y.T)


def cosine_distance(x, y):
    cosin = x @ y.T
    a = np.sqrt((x * x).sum(axis=1))[:, np.newaxis]
    b = np.sqrt((y * y).sum(axis=1))[np.newaxis, :]
    cosin = cosin / (a * b)
    return np.ones_like(cosin) - cosin
