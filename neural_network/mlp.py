"""
Feedforward Neural Network (Multi-Layer Perceptron) from scratch.

Supports arbitrary hidden layer sizes, ReLU hidden activations, softmax output,
and mini-batch gradient descent with manual backpropagation.
"""

import numpy as np


class MLP:
    def __init__(self, layer_sizes, learning_rate=0.01, n_epochs=200, batch_size=32, random_state=42):
        """
        Parameters
        ----------
        layer_sizes : list[int]
            e.g. [n_features, 16, 8, n_classes]
        """
        self.layer_sizes = layer_sizes
        self.lr = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.rng = np.random.RandomState(random_state)
        self.weights = []
        self.biases = []
        self.loss_history = []
        self._init_params()

    def _init_params(self):
        # He initialization, good default for ReLU networks
        for i in range(len(self.layer_sizes) - 1):
            fan_in = self.layer_sizes[i]
            fan_out = self.layer_sizes[i + 1]
            w = self.rng.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)
            b = np.zeros((1, fan_out))
            self.weights.append(w)
            self.biases.append(b)

    @staticmethod
    def _relu(z):
        return np.maximum(0, z)

    @staticmethod
    def _relu_deriv(z):
        return (z > 0).astype(float)

    @staticmethod
    def _softmax(z):
        z_shift = z - np.max(z, axis=1, keepdims=True)
        exp = np.exp(z_shift)
        return exp / np.sum(exp, axis=1, keepdims=True)

    def _one_hot(self, y, n_classes):
        one_hot = np.zeros((y.size, n_classes))
        one_hot[np.arange(y.size), y] = 1
        return one_hot

    def _forward(self, X):
        activations = [X]
        zs = []
        a = X
        n_layers = len(self.weights)

        for i in range(n_layers):
            z = a @ self.weights[i] + self.biases[i]
            zs.append(z)
            if i == n_layers - 1:
                a = self._softmax(z)
            else:
                a = self._relu(z)
            activations.append(a)

        return activations, zs

    def _backward(self, activations, zs, y_one_hot):
        n_layers = len(self.weights)
        n_samples = y_one_hot.shape[0]
        grads_w = [None] * n_layers
        grads_b = [None] * n_layers

        # output layer: softmax + cross-entropy -> simple gradient
        delta = activations[-1] - y_one_hot

        for i in reversed(range(n_layers)):
            grads_w[i] = (activations[i].T @ delta) / n_samples
            grads_b[i] = np.sum(delta, axis=0, keepdims=True) / n_samples

            if i > 0:
                delta = (delta @ self.weights[i].T) * self._relu_deriv(zs[i - 1])

        return grads_w, grads_b

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        n_classes = self.layer_sizes[-1]
        y_one_hot_full = self._one_hot(y, n_classes)
        n_samples = X.shape[0]

        for epoch in range(self.n_epochs):
            perm = self.rng.permutation(n_samples)
            X_shuffled, y_shuffled = X[perm], y_one_hot_full[perm]

            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                activations, zs = self._forward(X_batch)
                grads_w, grads_b = self._backward(activations, zs, y_batch)

                for i in range(len(self.weights)):
                    self.weights[i] -= self.lr * grads_w[i]
                    self.biases[i] -= self.lr * grads_b[i]

            activations, _ = self._forward(X)
            probs = activations[-1]
            eps = 1e-15
            loss = -np.mean(np.sum(y_one_hot_full * np.log(np.clip(probs, eps, 1)), axis=1))
            self.loss_history.append(loss)

        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        activations, _ = self._forward(X)
        return activations[-1]

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def score(self, X, y):
        y = np.asarray(y)
        return np.mean(self.predict(X) == y)


if __name__ == "__main__":
    np.random.seed(7)
    X0 = np.random.randn(100, 2) + np.array([-2, -2])
    X1 = np.random.randn(100, 2) + np.array([2, 2])
    X2 = np.random.randn(100, 2) + np.array([2, -2])
    X = np.vstack([X0, X1, X2])
    y = np.array([0] * 100 + [1] * 100 + [2] * 100)

    model = MLP(layer_sizes=[2, 16, 8, 3], learning_rate=0.05, n_epochs=100)
    model.fit(X, y)
    print(f"Train accuracy: {model.score(X, y):.4f}")
    print(f"Final loss: {model.loss_history[-1]:.4f}")
