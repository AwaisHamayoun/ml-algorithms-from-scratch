"""
Runs every algorithm in this repo on small synthetic datasets so you can
sanity-check that everything works end to end.

Run from the repo root with:
    python examples/example_usage.py
"""

import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from regression.linear_regression import LinearRegression
from classification.logistic_regression import LogisticRegression
from classification.knn import KNN
from classification.decision_tree import DecisionTree
from classification.naive_bayes import GaussianNaiveBayes
from classification.svm import SVM
from clustering.kmeans import KMeans
from clustering.hierarchical_clustering import AgglomerativeClustering
from clustering.dbscan import DBSCAN
from clustering.gaussian_mixture import GaussianMixture
from dimensionality_reduction.pca import PCA
from neural_network.mlp import MLP
from ensemble.random_forest import RandomForest


def make_binary_classification_data(seed=0, n=100):
    rng = np.random.RandomState(seed)
    X0 = rng.randn(n, 2) + np.array([-2, -2])
    X1 = rng.randn(n, 2) + np.array([2, 2])
    X = np.vstack([X0, X1])
    y = np.array([0] * n + [1] * n)
    return X, y


def make_multiclass_data(seed=7, n=80):
    rng = np.random.RandomState(seed)
    X0 = rng.randn(n, 2) + np.array([-2, -2])
    X1 = rng.randn(n, 2) + np.array([2, 2])
    X2 = rng.randn(n, 2) + np.array([2, -2])
    X = np.vstack([X0, X1, X2])
    y = np.array([0] * n + [1] * n + [2] * n)
    return X, y


def main():
    print("=" * 60)
    print("Linear Regression")
    rng = np.random.RandomState(1)
    X = rng.rand(100, 1) * 10
    y = 3.5 * X.squeeze() + 7 + rng.randn(100) * 2
    lr = LinearRegression(n_iters=1000).fit(X, y)
    print(f"  R^2: {lr.score(X, y):.4f}")

    X_cls, y_cls = make_binary_classification_data()

    print("=" * 60)
    print("Logistic Regression")
    model = LogisticRegression().fit(X_cls, y_cls)
    print(f"  Accuracy: {model.score(X_cls, y_cls):.4f}")

    print("=" * 60)
    print("K-Nearest Neighbors")
    model = KNN(k=5).fit(X_cls, y_cls)
    print(f"  Accuracy: {model.score(X_cls, y_cls):.4f}")

    print("=" * 60)
    print("Decision Tree")
    model = DecisionTree(max_depth=5).fit(X_cls, y_cls)
    print(f"  Accuracy: {model.score(X_cls, y_cls):.4f}")

    print("=" * 60)
    print("Gaussian Naive Bayes")
    model = GaussianNaiveBayes().fit(X_cls, y_cls)
    print(f"  Accuracy: {model.score(X_cls, y_cls):.4f}")

    print("=" * 60)
    print("Support Vector Machine")
    model = SVM().fit(X_cls, y_cls)
    print(f"  Accuracy: {model.score(X_cls, y_cls):.4f}")

    print("=" * 60)
    print("Random Forest (Ensemble)")
    model = RandomForest(n_trees=15, max_depth=6).fit(X_cls, y_cls)
    print(f"  Accuracy: {model.score(X_cls, y_cls):.4f}")

    print("=" * 60)
    print("K-Means Clustering")
    model = KMeans(k=2, random_state=42).fit(X_cls)
    print(f"  Inertia: {model.inertia_:.4f}")

    print("=" * 60)
    print("Agglomerative Hierarchical Clustering")
    model = AgglomerativeClustering(n_clusters=2, linkage="average")
    labels = model.fit_predict(X_cls)
    print(f"  Cluster sizes: {np.bincount(labels)}")

    print("=" * 60)
    print("DBSCAN")
    model = DBSCAN(eps=0.8, min_samples=5)
    labels = model.fit_predict(X_cls)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"  Clusters found: {n_clusters}, noise points: {np.sum(labels == -1)}")

    print("=" * 60)
    print("Gaussian Mixture Model (EM)")
    model = GaussianMixture(n_components=2).fit(X_cls)
    labels = model.predict(X_cls)
    print(f"  Cluster sizes: {np.bincount(labels)}")
    print(f"  Final log-likelihood: {model.log_likelihood_history_[-1]:.3f}")

    print("=" * 60)
    print("PCA")
    X5d = rng.randn(100, 5)
    model = PCA(n_components=2).fit(X5d)
    print(f"  Explained variance ratio: {model.explained_variance_ratio_}")

    print("=" * 60)
    print("Neural Network (MLP)")
    X_multi, y_multi = make_multiclass_data()
    model = MLP(layer_sizes=[2, 16, 8, 3], n_epochs=100).fit(X_multi, y_multi)
    print(f"  Accuracy: {model.score(X_multi, y_multi):.4f}")

    print("=" * 60)
    print("All algorithms ran successfully.")


if __name__ == "__main__":
    main()
