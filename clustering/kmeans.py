"""
K-Means Clustering from scratch (Lloyd's algorithm + k-means++ init).
"""

import numpy as np


class KMeans:
    def __init__(self, k=3, max_iters=300, tol=1e-4, init="k-means++", random_state=None):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.init = init
        self.random_state = random_state
        self.centroids = None
        self.labels_ = None
        self.inertia_ = None

    def _init_centroids(self, X):
        rng = np.random.RandomState(self.random_state)
        n_samples = X.shape[0]

        if self.init == "random":
            idx = rng.choice(n_samples, self.k, replace=False)
            return X[idx].copy()

        # k-means++ initialization
        centroids = [X[rng.randint(n_samples)]]
        for _ in range(1, self.k):
            dist_sq = np.array([min(np.sum((x - c) ** 2) for c in centroids) for x in X])
            probs = dist_sq / dist_sq.sum()
            cumulative = np.cumsum(probs)
            r = rng.rand()
            idx = np.searchsorted(cumulative, r)
            centroids.append(X[idx])
        return np.array(centroids)

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.centroids = self._init_centroids(X)

        for _ in range(self.max_iters):
            distances = np.sqrt(((X[:, np.newaxis, :] - self.centroids[np.newaxis, :, :]) ** 2).sum(axis=2))
            labels = np.argmin(distances, axis=1)

            new_centroids = np.array([
                X[labels == j].mean(axis=0) if np.any(labels == j) else self.centroids[j]
                for j in range(self.k)
            ])

            shift = np.linalg.norm(new_centroids - self.centroids)
            self.centroids = new_centroids
            if shift < self.tol:
                break

        self.labels_ = labels
        distances = np.sqrt(((X - self.centroids[labels]) ** 2).sum(axis=1))
        self.inertia_ = np.sum(distances ** 2)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        distances = np.sqrt(((X[:, np.newaxis, :] - self.centroids[np.newaxis, :, :]) ** 2).sum(axis=2))
        return np.argmin(distances, axis=1)


if __name__ == "__main__":
    np.random.seed(5)
    X0 = np.random.randn(50, 2) + np.array([-4, -4])
    X1 = np.random.randn(50, 2) + np.array([4, 4])
    X2 = np.random.randn(50, 2) + np.array([4, -4])
    X = np.vstack([X0, X1, X2])

    model = KMeans(k=3, random_state=42)
    model.fit(X)
    print(f"Inertia: {model.inertia_:.3f}")
    print(f"Centroids:\n{model.centroids}")
