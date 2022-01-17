import numpy as np

from sklearn.model_selection import KFold, BaseCrossValidator
cv = KFold(n_splits=3)
X = np.zeros((60000, 784))
for train, test in cv.split(X):
    print("train")
    print(train)
    print("test")
    print(test)
