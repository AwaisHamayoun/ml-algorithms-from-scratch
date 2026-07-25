"""
Agglomerative Hierarchical Clustering from scratch.

Bottom-up clustering: starts with every point as its own cluster and
repeatedly merges the two closest clusters until only `n_clusters` remain.

Supports linkage strategies: "single", "complete", "average".
"""

import numpy as np


class AgglomerativeClustering:
    def __init__(self, n_clusters=2, linkage="average"):
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.labels_ = None
        self.merge_history_ = []  # useful for plotting a dendrogram

    @staticmethod
    def _pairwise_distances(X):
        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        return np.sqrt(np.sum(diff ** 2, axis=-1))

    def _cluster_distance(self, dist_matrix, members_a, members_b):
        dists = [dist_matrix[i, j] for i in members_a for j in members_b]
        if self.linkage == "single":
            return min(dists)
        if self.linkage == "complete":
            return max(dists)
        return sum(dists) / len(dists)  # average

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        n_samples = X.shape[0]
        dist_matrix = self._pairwise_distances(X)

        # each cluster starts as a list of point indices it contains
        clusters = {i: [i] for i in range(n_samples)}
        active_ids = list(clusters.keys())
        next_id = n_samples

        while len(active_ids) > self.n_clusters:
            best_pair = None
            best_dist = np.inf

            for a_idx in range(len(active_ids)):
                for b_idx in range(a_idx + 1, len(active_ids)):
                    a, b = active_ids[a_idx], active_ids[b_idx]
                    d = self._cluster_distance(dist_matrix, clusters[a], clusters[b])
                    if d < best_dist:
                        best_dist = d
                        best_pair = (a, b)

            a, b = best_pair
            merged_members = clusters[a] + clusters[b]
            clusters[next_id] = merged_members
            self.merge_history_.append((a, b, best_dist, len(merged_members)))

            del clusters[a]
            del clusters[b]
            active_ids = list(clusters.keys())
            next_id += 1

        # assign final cluster labels back to original points
        labels = np.zeros(n_samples, dtype=int)
        for label, (_, members) in enumerate(clusters.items()):
            for point_idx in members:
                labels[point_idx] = label

        self.labels_ = labels
        return self

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_


if __name__ == "__main__":
    np.random.seed(8)
    X0 = np.random.randn(20, 2) + np.array([-4, -4])
    X1 = np.random.randn(20, 2) + np.array([4, 4])
    X2 = np.random.randn(20, 2) + np.array([4, -4])
    X = np.vstack([X0, X1, X2])

    model = AgglomerativeClustering(n_clusters=3, linkage="average")
    labels = model.fit_predict(X)
    print(f"Cluster label counts: {np.bincount(labels)}")
