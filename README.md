# ML Algorithms From Scratch

Implementations of core machine learning algorithms built from scratch using only **NumPy** — no scikit-learn, no black boxes. Every model is written to be readable, mathematically transparent, and easy to compare against library implementations.

## Why this repo

Most ML libraries hide the math behind `.fit()` and `.predict()`. This repo does the opposite: every gradient, every loss function, and every update rule is written out explicitly, so you can see exactly what's happening under the hood.

## Algorithms Implemented

This repo covers all three major branches of classical ML — **supervised**, **unsupervised**, and **ensemble** methods — plus a from-scratch neural network.

### Supervised Learning

| Category | Algorithm | File |
|---|---|---|
| Regression | Linear Regression (Gradient Descent + Normal Equation) | [`regression/linear_regression.py`](regression/linear_regression.py) |
| Classification | Logistic Regression | [`classification/logistic_regression.py`](classification/logistic_regression.py) |
| Classification | K-Nearest Neighbors | [`classification/knn.py`](classification/knn.py) |
| Classification | Decision Tree (CART, Gini impurity, built recursively) | [`classification/decision_tree.py`](classification/decision_tree.py) |
| Classification | Gaussian Naive Bayes | [`classification/naive_bayes.py`](classification/naive_bayes.py) |
| Classification | Support Vector Machine (hinge loss, sub-gradient descent) | [`classification/svm.py`](classification/svm.py) |

### Unsupervised Learning

| Category | Algorithm | File |
|---|---|---|
| Clustering | K-Means (k-means++ initialization) | [`clustering/kmeans.py`](clustering/kmeans.py) |
| Clustering | Agglomerative Hierarchical Clustering (single/complete/average linkage) | [`clustering/hierarchical_clustering.py`](clustering/hierarchical_clustering.py) |
| Clustering | DBSCAN (density-based, handles noise + arbitrary shapes) | [`clustering/dbscan.py`](clustering/dbscan.py) |
| Clustering | Gaussian Mixture Model (Expectation-Maximization) | [`clustering/gaussian_mixture.py`](clustering/gaussian_mixture.py) |
| Dimensionality Reduction | PCA (eigendecomposition) | [`dimensionality_reduction/pca.py`](dimensionality_reduction/pca.py) |

### Ensemble Methods

| Category | Algorithm | File |
|---|---|---|
| Ensemble | Random Forest (bagged, recursively-built Decision Trees) | [`ensemble/random_forest.py`](ensemble/random_forest.py) |

### Neural Networks

| Category | Algorithm | File |
|---|---|---|
| Deep Learning | Feedforward MLP (manual backprop, ReLU + softmax, mini-batch GD) | [`neural_network/mlp.py`](neural_network/mlp.py) |

## Project Structure

```
ml-algorithms-from-scratch/
├── regression/
│   └── linear_regression.py
├── classification/
│   ├── logistic_regression.py
│   ├── knn.py
│   ├── decision_tree.py
│   ├── naive_bayes.py
│   └── svm.py
├── clustering/
│   ├── kmeans.py
│   ├── hierarchical_clustering.py
│   ├── dbscan.py
│   └── gaussian_mixture.py
├── dimensionality_reduction/
│   └── pca.py
├── ensemble/
│   └── random_forest.py
├── neural_network/
│   └── mlp.py
├── examples/
│   └── example_usage.py
├── tests/
│   └── test_algorithms.py
├── requirements.txt
├── LICENSE
└── README.md
```

## Getting Started

```bash
git clone https://github.com/<your-username>/ml-algorithms-from-scratch.git
cd ml-algorithms-from-scratch
pip install -r requirements.txt
```

Run every algorithm on synthetic data to confirm everything works:

```bash
python examples/example_usage.py
```

Run the test suite:

```bash
python -m pytest tests/ -v
```

## Usage Example

Each model follows a scikit-learn-style API (`fit`, `predict`, `score`), so they're easy to drop into existing workflows or compare directly against scikit-learn.

```python
from classification.logistic_regression import LogisticRegression
import numpy as np

X = np.random.randn(200, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

model = LogisticRegression(learning_rate=0.1, n_iters=1000)
model.fit(X, y)

print("Accuracy:", model.score(X, y))
```

## Design Principles

- **No external ML dependencies.** Only NumPy is used for numerical operations — every algorithm is implemented from first principles.
- **Consistent API.** Every model exposes `fit()`, `predict()`, and `score()`, mirroring the scikit-learn interface for easy comparison.
- **Readable over clever.** Code favors clarity over micro-optimization, since the goal is to learn and demonstrate the underlying math.
- **Runnable examples.** Every file has a `if __name__ == "__main__":` block that runs the algorithm on synthetic data.

## Roadmap

- [ ] AdaBoost / Gradient Boosting
- [ ] Linear Discriminant Analysis
- [ ] Convolutional Neural Network (from scratch)
- [ ] Ridge / Lasso Regression
- [ ] t-SNE
- [ ] Recurrent Neural Network / LSTM (from scratch)

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
## Certifications

This repo was built as part of an ongoing, hands-on complement to formal coursework:

- **[Machine Learning Specialization](https://coursera.org/verify/specialization/1M070BP1PHU0)** — DeepLearning.AI & Stanford Online (Andrew Ng), 3      courses
  - Supervised Machine Learning: Regression and Classification
  - [Advanced Learning Algorithms](https://coursera.org/verify/I67K7O5GEHTM)
  - [Unsupervised Learning, Recommenders, Reinforcement Learning](https://coursera.org/verify/NLK3I0SII701)

## Author

Built by Awais, BS AI student, as part of an ongoing portfolio of ML/AI engineering projects.
