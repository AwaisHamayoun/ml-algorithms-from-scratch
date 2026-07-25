"""
Principal Component Analysis from scratch (via eigendecomposition of the covariance matrix).
"""

import numpy as np


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.explained_variance_ratio_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mean = X.mean(axis=0)
        X_centered = X - self.mean

        cov = np.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)  # eigh -> ascending order, symmetric matrix

        # sort descending
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[:, idxs]

        self.components = eigenvectors[:, : self.n_components].T
        total_var = eigenvalues.sum()
        self.explained_variance_ratio_ = eigenvalues[: self.n_components] / total_var

        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        X_centered = X - self.mean
        return X_centered @ self.components.T

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_transformed):
        return X_transformed @ self.components + self.mean


if __name__ == "__main__":
    np.random.seed(6)
    X = np.random.randn(100, 5)
    X[:, 1] = X[:, 0] * 2 + np.random.randn(100) * 0.1  # correlated feature

    model = PCA(n_components=2)
    X_reduced = model.fit_transform(X)
    print(f"Reduced shape: {X_reduced.shape}")
    print(f"Explained variance ratio: {model.explained_variance_ratio_}")
