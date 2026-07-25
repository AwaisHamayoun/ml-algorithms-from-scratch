"""
Basic sanity tests. Run with:
    python -m pytest tests/ -v
or simply:
    python tests/test_algorithms.py
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


def _binary_data(seed=0, n=60):
    rng = np.random.RandomState(seed)
    X0 = rng.randn(n, 2) + np.array([-2, -2])
    X1 = rng.randn(n, 2) + np.array([2, 2])
    X = np.vstack([X0, X1])
    y = np.array([0] * n + [1] * n)
    return X, y


def test_linear_regression():
    rng = np.random.RandomState(1)
    X = rng.rand(50, 1) * 10
    y = 3 * X.squeeze() + 5
    model = LinearRegression(n_iters=2000, learning_rate=0.01).fit(X, y)
    assert model.score(X, y) > 0.95


def test_logistic_regression():
    X, y = _binary_data()
    model = LogisticRegression(n_iters=1000).fit(X, y)
    assert model.score(X, y) > 0.9


def test_knn():
    X, y = _binary_data()
    model = KNN(k=5).fit(X, y)
    assert model.score(X, y) > 0.9


def test_decision_tree():
    X, y = _binary_data()
    model = DecisionTree(max_depth=5).fit(X, y)
    assert model.score(X, y) > 0.9


def test_naive_bayes():
    X, y = _binary_data()
    model = GaussianNaiveBayes().fit(X, y)
    assert model.score(X, y) > 0.9


def test_svm():
    X, y = _binary_data()
    model = SVM(n_iters=500).fit(X, y)
    assert model.score(X, y) > 0.9


def test_kmeans():
    X, _ = _binary_data()
    model = KMeans(k=2, random_state=42).fit(X)
    assert model.centroids.shape == (2, 2)
    assert model.inertia_ > 0


def test_pca():
    rng = np.random.RandomState(2)
    X = rng.randn(50, 5)
    model = PCA(n_components=2).fit(X)
    X_t = model.transform(X)
    assert X_t.shape == (50, 2)


def test_mlp():
    X, y = _binary_data()
    model = MLP(layer_sizes=[2, 8, 2], n_epochs=50).fit(X, y)
    assert model.score(X, y) > 0.8


def test_random_forest():
    X, y = _binary_data()
    model = RandomForest(n_trees=10, max_depth=6).fit(X, y)
    assert model.score(X, y) > 0.9


def test_hierarchical_clustering():
    X, _ = _binary_data()
    model = AgglomerativeClustering(n_clusters=2)
    labels = model.fit_predict(X)
    assert len(np.unique(labels)) == 2


def test_dbscan():
    X, _ = _binary_data()
    model = DBSCAN(eps=0.8, min_samples=5)
    labels = model.fit_predict(X)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    assert n_clusters >= 1


def test_gaussian_mixture():
    X, _ = _binary_data()
    model = GaussianMixture(n_components=2).fit(X)
    labels = model.predict(X)
    assert len(np.unique(labels)) <= 2
    assert len(model.log_likelihood_history_) > 0


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"PASS: {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL: {t.__name__} - {e}")
    print(f"\n{passed}/{len(tests)} tests passed.")
