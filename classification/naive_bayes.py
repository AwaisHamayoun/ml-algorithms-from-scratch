"""
Gaussian Naive Bayes from scratch.

Assumes each feature is normally distributed within each class.
P(y | x) proportional to P(y) * prod_i P(x_i | y)
"""

import numpy as np


class GaussianNaiveBayes:
    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing
        self.classes = None
        self.mean = {}
        self.var = {}
        self.priors = {}

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes = np.unique(y)

        for c in self.classes:
            X_c = X[y == c]
            self.mean[c] = X_c.mean(axis=0)
            self.var[c] = X_c.var(axis=0) + self.var_smoothing
            self.priors[c] = X_c.shape[0] / X.shape[0]

        return self

    def _gaussian_log_prob(self, x, mean, var):
        # log of Gaussian PDF, summed over features (independence assumption)
        log_prob = -0.5 * np.log(2 * np.pi * var) - ((x - mean) ** 2) / (2 * var)
        return np.sum(log_prob)

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        preds = []
        for x in X:
            posteriors = []
            for c in self.classes:
                log_prior = np.log(self.priors[c])
                log_likelihood = self._gaussian_log_prob(x, self.mean[c], self.var[c])
                posteriors.append(log_prior + log_likelihood)
            preds.append(self.classes[np.argmax(posteriors)])
        return np.array(preds)

    def score(self, X, y):
        y = np.asarray(y)
        return np.mean(self.predict(X) == y)


if __name__ == "__main__":
    np.random.seed(3)
    X0 = np.random.randn(50, 2) + np.array([-2, -2])
    X1 = np.random.randn(50, 2) + np.array([2, 2])
    X = np.vstack([X0, X1])
    y = np.array([0] * 50 + [1] * 50)

    model = GaussianNaiveBayes()
    model.fit(X, y)
    print(f"Train accuracy: {model.score(X, y):.4f}")
