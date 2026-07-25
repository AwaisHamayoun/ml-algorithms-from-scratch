"""
K-Nearest Neighbors from scratch (Classification + Regression).
No training phase - stores data and computes distances at predict time.
"""

import numpy as np
from collections import Counter


class KNN:
    def __init__(self, k=5, task="classification", distance="euclidean"):
        """
        Parameters
        ----------
        k : int
            Number of neighbors to consider.
        task : str
            "classification" or "regression".
        distance : str
            "euclidean" or "manhattan".
        """
        self.k = k
        self.task = task
        self.distance = distance
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y)
        return self

    def _compute_distances(self, x):
        if self.distance == "manhattan":
            return np.sum(np.abs(self.X_train - x), axis=1)
        return np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))

    def _predict_one(self, x):
        distances = self._compute_distances(x)
        k_idx = np.argsort(distances)[: self.k]
        k_labels = self.y_train[k_idx]

        if self.task == "regression":
            return np.mean(k_labels)

        most_common = Counter(k_labels).most_common(1)
        return most_common[0][0]

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_one(x) for x in X])

    def score(self, X, y):
        y = np.asarray(y)
        preds = self.predict(X)
        if self.task == "regression":
            ss_res = np.sum((y - preds) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            return 1 - ss_res / ss_tot
        return np.mean(preds == y)


if __name__ == "__main__":
    np.random.seed(1)
    X0 = np.random.randn(50, 2) + np.array([-2, -2])
    X1 = np.random.randn(50, 2) + np.array([2, 2])
    X = np.vstack([X0, X1])
    y = np.array([0] * 50 + [1] * 50)

    model = KNN(k=5)
    model.fit(X, y)
    print(f"Train accuracy: {model.score(X, y):.4f}")
