"""
DBSCAN (Density-Based Spatial Clustering of Applications with Noise) from scratch.

Groups together points that are closely packed (density-reachable), and marks
low-density outlier points as noise. Unlike K-Means, it does NOT require
specifying the number of clusters in advance and can find arbitrarily shaped clusters.

Cluster labels: 0, 1, 2, ... for clusters, -1 for noise points.
"""

import numpy as np


class DBSCAN:
    def __init__(self, eps=0.5, min_samples=5):
        """
        Parameters
        ----------
        eps : float
            Maximum distance between two samples for them to be considered neighbors.
        min_samples : int
            Minimum number of points required to form a dense region (core point).
        """
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None

    def _region_query(self, X, point_idx):
        distances = np.sqrt(np.sum((X - X[point_idx]) ** 2, axis=1))
        return np.where(distances <= self.eps)[0]

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        n_samples = X.shape[0]
        labels = np.full(n_samples, -2, dtype=int)  # -2 = unvisited, -1 = noise
        cluster_id = 0

        for point_idx in range(n_samples):
            if labels[point_idx] != -2:
                continue  # already processed

            neighbors = self._region_query(X, point_idx)

            if len(neighbors) < self.min_samples:
                labels[point_idx] = -1  # mark as noise (may be re-labeled as border point later)
                continue

            # start a new cluster
            labels[point_idx] = cluster_id
            seed_set = list(neighbors)
            i = 0
            while i < len(seed_set):
                neighbor_idx = seed_set[i]

                if labels[neighbor_idx] == -1:
                    labels[neighbor_idx] = cluster_id  # noise point becomes a border point

                if labels[neighbor_idx] == -2:
                    labels[neighbor_idx] = cluster_id
                    neighbor_neighbors = self._region_query(X, neighbor_idx)
                    if len(neighbor_neighbors) >= self.min_samples:
                        seed_set.extend(neighbor_neighbors)

                i += 1

            cluster_id += 1

        self.labels_ = labels
        return self

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_


if __name__ == "__main__":
    np.random.seed(9)
    X0 = np.random.randn(40, 2) * 0.5 + np.array([-3, -3])
    X1 = np.random.randn(40, 2) * 0.5 + np.array([3, 3])
    noise = np.random.uniform(-8, 8, size=(5, 2))
    X = np.vstack([X0, X1, noise])

    model = DBSCAN(eps=1.0, min_samples=5)
    labels = model.fit_predict(X)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"Clusters found: {n_clusters}, noise points: {np.sum(labels == -1)}")
