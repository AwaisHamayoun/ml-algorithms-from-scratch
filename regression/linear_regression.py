"""
Linear Regression from scratch (Batch Gradient Descent + Normal Equation).

Model:  y_hat = X @ w + b
Loss:   Mean Squared Error
"""

import numpy as np


class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iters=1000, method="gradient_descent"):
        """
        Parameters
        ----------
        learning_rate : float
            Step size for gradient descent.
        n_iters : int
            Number of gradient descent iterations.
        method : str
            "gradient_descent" or "normal_equation".
        """
        self.lr = learning_rate
        self.n_iters = n_iters
        self.method = method
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        n_samples, n_features = X.shape

        if self.method == "normal_equation":
            X_b = np.hstack([np.ones((n_samples, 1)), X])
            # theta = (X^T X)^-1 X^T y   (pseudo-inverse for numerical stability)
            theta = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y
            self.bias = theta[0]
            self.weights = theta[1:]
            return self

        # Gradient descent
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.n_iters):
            y_pred = X @ self.weights + self.bias
            error = y_pred - y

            dw = (2 / n_samples) * (X.T @ error)
            db = (2 / n_samples) * np.sum(error)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            loss = np.mean(error ** 2)
            self.loss_history.append(loss)

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return X @ self.weights + self.bias

    def score(self, X, y):
        """R^2 coefficient of determination."""
        y = np.asarray(y, dtype=float).reshape(-1)
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot


if __name__ == "__main__":
    np.random.seed(42)
    X = np.random.rand(100, 1) * 10
    y = 3.5 * X.squeeze() + 7 + np.random.randn(100) * 2

    model = LinearRegression(learning_rate=0.01, n_iters=2000)
    model.fit(X, y)
    print(f"Weights: {model.weights}, Bias: {model.bias:.3f}")
    print(f"R^2 score: {model.score(X, y):.4f}")
