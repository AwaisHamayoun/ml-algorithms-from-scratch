"""
Linear Support Vector Machine from scratch.

Trained with sub-gradient descent on the hinge-loss objective:
    L = lambda * ||w||^2 + (1/n) * sum(max(0, 1 - y_i (w.x_i - b)))
Labels are expected to be -1 / +1.
"""

import numpy as np


class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y_ = np.where(np.asarray(y) <= 0, -1, 1)  # convert {0,1} -> {-1,1} if needed
        n_samples, n_features = X.shape

        self.w = np.zeros(n_features)
        self.b = 0.0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    dw = 2 * self.lambda_param * self.w
                    db = 0
                else:
                    dw = 2 * self.lambda_param * self.w - y_[idx] * x_i
                    db = y_[idx]

                self.w -= self.lr * dw
                self.b -= self.lr * db

        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        return X @ self.w - self.b

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1, -1)

    def score(self, X, y):
        y_ = np.where(np.asarray(y) <= 0, -1, 1)
        return np.mean(self.predict(X) == y_)


if __name__ == "__main__":
    np.random.seed(4)
    X0 = np.random.randn(50, 2) + np.array([-2, -2])
    X1 = np.random.randn(50, 2) + np.array([2, 2])
    X = np.vstack([X0, X1])
    y = np.array([0] * 50 + [1] * 50)  # will be mapped to -1/+1 internally

    model = SVM()
    model.fit(X, y)
    print(f"Train accuracy: {model.score(X, y):.4f}")
