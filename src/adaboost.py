'''
https://claude.site/artifacts/51cda0c3-0474-44a2-8c83-8b69f2babbb1
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier


class SimpleAdaBoost:
    def __init__(self, n_estimators=3):
        self.n_estimators = n_estimators
        self.weak_classifiers = []
        self.alphas = []

    def _create_weak_learner(self):
        # Using a decision stump (depth-1 tree) as weak learner
        return DecisionTreeClassifier(max_depth=1)

    def fit(self, X, y):
        n_samples = len(X)
        # Initialize weights uniformly
        weights = np.ones(n_samples) / n_samples

        # Store predictions and weights from each stage for visualization
        self.stage_predictions = np.zeros((self.n_estimators, n_samples))
        self.stage_weights = np.zeros((self.n_estimators, n_samples))

        for i in range(self.n_estimators):
            # Train weak learner
            weak_learner = self._create_weak_learner()
            weak_learner.fit(X, y, sample_weight=weights)

            # Get predictions
            predictions = weak_learner.predict(X)

            # Calculate weighted error
            incorrect = predictions != y
            error = np.sum(weights * incorrect) / np.sum(weights)

            # Calculate alpha (importance of this classifier)
            alpha = 0.5 * np.log((1 - error) / (error + 1e-10))

            # Update weights
            weights *= np.exp(-alpha * y * predictions)
            weights /= np.sum(weights)  # Normalize

            # Store the classifier and its importance
            self.weak_classifiers.append(weak_learner)
            self.alphas.append(alpha)

            # Store stage predictions and weights for visualization
            self.stage_predictions[i] = predictions
            self.stage_weights[i] = weights.copy()

    def predict(self, X):
        # Weighted sum of weak classifier predictions
        predictions = np.zeros(len(X))
        for alpha, clf in zip(self.alphas, self.weak_classifiers):
            predictions += alpha * clf.predict(X)
        return np.sign(predictions)

def generate_complex_data(n_samples=200):
    """Generate a more complex, non-linearly separable dataset"""
    np.random.seed(42)

    # Generate several gaussian clusters for both classes
    X = []
    y = []

    # Class -1: Two clusters
    X1 = np.random.randn(n_samples//4, 2) * 0.5 + [-2, -2]
    X2 = np.random.randn(n_samples//4, 2) * 0.5 + [2, 2]
    X.extend([X1, X2])
    y.extend([-1] * (n_samples//2))

    # Class 1: Three clusters in between
    X3 = np.random.randn(n_samples//6, 2) * 0.3 + [0, 0]
    X4 = np.random.randn(n_samples//6, 2) * 0.3 + [-1, 1]
    X5 = np.random.randn(n_samples//6, 2) * 0.3 + [1, -1]
    X.extend([X3, X4, X5])
    y.extend([1] * (n_samples//2))

    # Add some noise points
    noise_points = np.random.uniform(-3, 3, (n_samples//10, 2))
    noise_labels = np.random.choice([-1, 1], size=n_samples//10)
    X.append(noise_points)
    y.extend(noise_labels)

    X = np.vstack(X)
    y = np.array(y)

    # Shuffle the data
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]

def visualize_adaboost_stages(X, y, model, title="AdaBoost Learning Stages"):
    # Create a grid for decision boundary visualization
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                        np.arange(y_min, y_max, 0.02))

    # Plot stages
    fig = plt.figure(figsize=(20, 5))

    for i in range(model.n_estimators):
        plt.subplot(1, model.n_estimators, i+1)

        # Plot the decision boundary for this stage
        Z = model.weak_classifiers[i].predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')

        # Plot the points with size proportional to their weights
        weights = model.stage_weights[i]
        sizes = 50 * (weights / weights.max())

        plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1],
                   c='red', s=sizes[y == -1], alpha=0.6, label='Class -1')
        plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1],
                   c='blue', s=sizes[y == 1], alpha=0.6, label='Class 1')

        plt.title(f'Weak Learner {i+1}\nα = {model.alphas[i]:.2f}')
        plt.legend()

    plt.tight_layout()
    plt.show()

    # Plot final ensemble decision boundary
    plt.figure(figsize=(8, 6))

    # Create a mesh of points to plot the decision boundary
    Z = np.zeros(xx.ravel().shape)
    for alpha, clf in zip(model.alphas, model.weak_classifiers):
        Z += alpha * clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = np.sign(Z)
    Z = Z.reshape(xx.shape)

    # Plot the decision boundary
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], c='red', label='Class -1')
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c='blue', label='Class 1')
    plt.title('Final Ensemble Decision Boundary')
    plt.legend()
    plt.show()

# Demo usage
if __name__ == "__main__":
    # Generate complex demo data
    X, y = generate_complex_data()

    # Create and train AdaBoost model
    model = SimpleAdaBoost(n_estimators=3)
    model.fit(X, y)

    # Visualize the learning stages
    visualize_adaboost_stages(X, y, model)