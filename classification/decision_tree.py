"""
Decision Tree Classifier from scratch (CART-style, Gini impurity, recursive splitting).
"""

import numpy as np
from collections import Counter


class _Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # set only for leaf nodes

    def is_leaf(self):
        return self.value is not None


class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2, n_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features  # for random-forest-style feature subsampling
        self.root = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.n_features = X.shape[1] if self.n_features is None else min(self.n_features, X.shape[1])
        self.root = self._grow_tree(X, y)
        return self

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            return _Node(value=self._most_common_label(y))

        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)

        best_feat, best_thresh = self._best_split(X, y, feat_idxs)

        if best_feat is None:
            return _Node(value=self._most_common_label(y))

        left_idxs = X[:, best_feat] <= best_thresh
        right_idxs = ~left_idxs

        left = self._grow_tree(X[left_idxs], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs], y[right_idxs], depth + 1)
        return _Node(feature=best_feat, threshold=best_thresh, left=left, right=right)

    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None

        parent_gini = self._gini(y)

        for feat in feat_idxs:
            thresholds = np.unique(X[:, feat])
            for thresh in thresholds:
                left_mask = X[:, feat] <= thresh
                right_mask = ~left_mask
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue

                n = len(y)
                n_l, n_r = left_mask.sum(), right_mask.sum()
                gini_l = self._gini(y[left_mask])
                gini_r = self._gini(y[right_mask])
                weighted_gini = (n_l / n) * gini_l + (n_r / n) * gini_r
                gain = parent_gini - weighted_gini

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat
                    split_thresh = thresh

        return split_idx, split_thresh

    @staticmethod
    def _gini(y):
        counts = np.bincount(y.astype(int)) if y.dtype != object else np.array(list(Counter(y).values()))
        probs = counts / len(y)
        return 1 - np.sum(probs ** 2)

    @staticmethod
    def _most_common_label(y):
        return Counter(y).most_common(1)[0][0]

    def _traverse(self, x, node):
        if node.is_leaf():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([self._traverse(x, self.root) for x in X])

    def score(self, X, y):
        y = np.asarray(y)
        return np.mean(self.predict(X) == y)


if __name__ == "__main__":
    np.random.seed(2)
    X0 = np.random.randn(50, 2) + np.array([-2, -2])
    X1 = np.random.randn(50, 2) + np.array([2, 2])
    X = np.vstack([X0, X1])
    y = np.array([0] * 50 + [1] * 50)

    model = DecisionTree(max_depth=5)
    model.fit(X, y)
    print(f"Train accuracy: {model.score(X, y):.4f}")
