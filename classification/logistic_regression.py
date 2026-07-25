"""
Logistic Regression from scratch (Binary Classification, Gradient Descent).

Model:  p = sigmoid(X @ w + b)
Loss:   Binary Cross-Entropy
"""

import numpy as np


class LogisticRegression:
    def __init__(self, learning_rate=0.1, n_iters=1000, l2_lambda=0.0):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.l2_lambda = l2_lambda
        self.weights = None
        self.bias = None
        self.loss_history = []

    @staticmethod
    def _sigmoid(z):
        z = np.clip(z, -500, 500)  # avoid overflow
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.n_iters):
            linear_out = X @ self.weights + self.bias
            y_pred = self._sigmoid(linear_out)
            error = y_pred - y

            dw = (1 / n_samples) * (X.T @ error) + (self.l2_lambda / n_samples) * self.weights
            db = (1 / n_samples) * np.sum(error)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            eps = 1e-15
            y_pred_c = np.clip(y_pred, eps, 1 - eps)
            loss = -np.mean(y * np.log(y_pred_c) + (1 - y) * np.log(1 - y_pred_c))
            self.loss_history.append(loss)

        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        return self._sigmoid(X @ self.weights + self.bias)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def score(self, X, y):
        y = np.asarray(y).reshape(-1)
        preds = self.predict(X)
        return np.mean(preds == y)


if __name__ == "__main__":
    np.random.seed(0)
    X0 = np.random.randn(100, 2) + np.array([-2, -2])
    X1 = np.random.randn(100, 2) + np.array([2, 2])
    X = np.vstack([X0, X1])
    y = np.array([0] * 100 + [1] * 100)

    model = LogisticRegression(learning_rate=0.1, n_iters=1000)
    model.fit(X, y)
    print(f"Train accuracy: {model.score(X, y):.4f}")
