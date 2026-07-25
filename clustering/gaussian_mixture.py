"""
Gaussian Mixture Model from scratch, fit with the Expectation-Maximization (EM) algorithm.

Unlike K-Means (hard cluster assignment), GMM gives soft, probabilistic
cluster membership and models each cluster as its own Gaussian distribution
(with its own mean, covariance, and mixing weight).
"""

import numpy as np


class GaussianMixture:
    def __init__(self, n_components=3, max_iters=100, tol=1e-4, random_state=42):
        self.n_components = n_components
        self.max_iters = max_iters
        self.tol = tol
        self.rng = np.random.RandomState(random_state)

        self.means_ = None
        self.covariances_ = None
        self.weights_ = None
        self.log_likelihood_history_ = []

    def _init_params(self, X):
        n_samples, n_features = X.shape
        idx = self.rng.choice(n_samples, self.n_components, replace=False)
        self.means_ = X[idx].copy()
        self.covariances_ = np.array([np.eye(n_features) for _ in range(self.n_components)])
        self.weights_ = np.full(self.n_components, 1 / self.n_components)

    @staticmethod
    def _multivariate_gaussian_pdf(X, mean, cov):
        n_features = X.shape[1]
        cov_reg = cov + np.eye(n_features) * 1e-6  # regularize for numerical stability
        diff = X - mean
        inv_cov = np.linalg.inv(cov_reg)
        det_cov = np.linalg.det(cov_reg)

        exponent = -0.5 * np.sum(diff @ inv_cov * diff, axis=1)
        norm_const = 1 / np.sqrt(((2 * np.pi) ** n_features) * det_cov)
        return norm_const * np.exp(exponent)

    def _e_step(self, X):
        n_samples = X.shape[0]
        resp = np.zeros((n_samples, self.n_components))

        for k in range(self.n_components):
            resp[:, k] = self.weights_[k] * self._multivariate_gaussian_pdf(
                X, self.means_[k], self.covariances_[k]
            )

        total_prob = resp.sum(axis=1, keepdims=True)
        total_prob[total_prob == 0] = 1e-10
        resp = resp / total_prob
        log_likelihood = np.sum(np.log(total_prob.squeeze() + 1e-10))
        return resp, log_likelihood

    def _m_step(self, X, resp):
        n_samples, n_features = X.shape
        Nk = resp.sum(axis=0)  # effective number of points per component

        for k in range(self.n_components):
            self.weights_[k] = Nk[k] / n_samples
            self.means_[k] = (resp[:, k, np.newaxis] * X).sum(axis=0) / Nk[k]
            diff = X - self.means_[k]
            weighted_cov = (resp[:, k, np.newaxis] * diff).T @ diff / Nk[k]
            self.covariances_[k] = weighted_cov

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self._init_params(X)

        prev_log_likelihood = -np.inf
        for _ in range(self.max_iters):
            resp, log_likelihood = self._e_step(X)
            self._m_step(X, resp)
            self.log_likelihood_history_.append(log_likelihood)

            if abs(log_likelihood - prev_log_likelihood) < self.tol:
                break
            prev_log_likelihood = log_likelihood

        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        resp, _ = self._e_step(X)
        return resp

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


if __name__ == "__main__":
    np.random.seed(10)
    X0 = np.random.randn(60, 2) + np.array([-3, -3])
    X1 = np.random.randn(60, 2) + np.array([3, 3])
    X2 = np.random.randn(60, 2) + np.array([3, -3])
    X = np.vstack([X0, X1, X2])

    model = GaussianMixture(n_components=3)
    model.fit(X)
    labels = model.predict(X)
    print(f"Cluster label counts: {np.bincount(labels)}")
    print(f"Final log-likelihood: {model.log_likelihood_history_[-1]:.3f}")
