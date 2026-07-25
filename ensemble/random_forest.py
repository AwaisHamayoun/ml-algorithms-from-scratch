"""
Random Forest Classifier from scratch.

An ensemble of Decision Trees (which are themselves built via recursion),
each trained on a bootstrap sample of the data with a random subset of
features considered at every split. Final prediction is by majority vote.
"""

import sys
import os
import numpy as np
from collections import Counter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from classification.decision_tree import DecisionTree


class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2,
                 n_features=None, random_state=42):
        """
        Parameters
        ----------
        n_trees : int
            Number of trees in the forest.
        max_depth : int
            Max depth for each individual tree.
        n_features : int or None
            Number of features to randomly consider at each split.
            Defaults to sqrt(total_features), the standard Random Forest heuristic.
        """
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.rng = np.random.RandomState(random_state)
        self.trees = []

    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        idxs = self.rng.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        n_total_features = X.shape[1]

        feat_count = self.n_features or max(1, int(np.sqrt(n_total_features)))

        self.trees = []
        for _ in range(self.n_trees):
            X_sample, y_sample = self._bootstrap_sample(X, y)
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_features=feat_count,
            )
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        # shape: (n_trees, n_samples)
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        # transpose -> (n_samples, n_trees), majority vote per sample
        tree_preds = tree_preds.T
        return np.array([Counter(row).most_common(1)[0][0] for row in tree_preds])

    def score(self, X, y):
        y = np.asarray(y)
        return np.mean(self.predict(X) == y)


if __name__ == "__main__":
    np.random.seed(11)
    X0 = np.random.randn(80, 2) + np.array([-2, -2])
    X1 = np.random.randn(80, 2) + np.array([2, 2])
    X = np.vstack([X0, X1])
    y = np.array([0] * 80 + [1] * 80)

    model = RandomForest(n_trees=15, max_depth=6)
    model.fit(X, y)
    print(f"Train accuracy: {model.score(X, y):.4f}")
