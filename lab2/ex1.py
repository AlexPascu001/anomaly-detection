import numpy as np
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('TkAgg')

a, b, c = 2, 3, 5
n_points = 100
mu_values = [0, 0.5, 1]
sigma2_values = [0.1, 1, 2]


def compute_leverage_scores(X):
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    leverage_scores = np.sum(U ** 2, axis=1)
    return leverage_scores


def plot_with_leverage_scores(X, y, leverage_scores, title, ax):
    ax.scatter(X[:, 1], y, color="blue", label="Data Points")
    high_leverage_points = leverage_scores > X.shape[1] / n_points
    ax.scatter(X[high_leverage_points, 1], y[high_leverage_points], color="red", label="High Leverage")
    ax.set_title(title)
    ax.legend()


fig, axes = plt.subplots(len(mu_values), 4, figsize=(20, 10))
fig.suptitle("Leverage Scores with Different Variances on X and Y")

for i, (mu, sigma2) in enumerate(zip(mu_values, sigma2_values)):
    x1 = np.random.normal(mu, np.sqrt(0.1), n_points)
    y = a * x1 + b + np.random.normal(mu, np.sqrt(0.1), n_points)
    X = np.column_stack((np.ones(n_points), x1))
    leverage_scores = compute_leverage_scores(X)
    plot_with_leverage_scores(X, y, leverage_scores, f"mu={mu}, small variance on X and Y", axes[i, 0])

    x1 = np.random.normal(mu, np.sqrt(sigma2), n_points)
    y = a * x1 + b + np.random.normal(mu, np.sqrt(0.1), n_points)
    X = np.column_stack((np.ones(n_points), x1))
    leverage_scores = compute_leverage_scores(X)
    plot_with_leverage_scores(X, y, leverage_scores, f"mu={mu}, high variance on X", axes[i, 1])

    x1 = np.random.normal(mu, np.sqrt(0.1), n_points)
    y = a * x1 + b + np.random.normal(mu, np.sqrt(sigma2), n_points)
    X = np.column_stack((np.ones(n_points), x1))
    leverage_scores = compute_leverage_scores(X)
    plot_with_leverage_scores(X, y, leverage_scores, f"mu={mu}, high variance on Y", axes[i, 2])

    x1 = np.random.normal(mu, np.sqrt(sigma2), n_points)
    y = a * x1 + b + np.random.normal(mu, np.sqrt(sigma2), n_points)
    X = np.column_stack((np.ones(n_points), x1))
    leverage_scores = compute_leverage_scores(X)
    plot_with_leverage_scores(X, y, leverage_scores, f"mu={mu}, high variance on both X and Y", axes[i, 3])

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.show()

for mu, sigma2 in zip(mu_values, sigma2_values):
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(f"3D Leverage Scores with Different Variances on X and Y (mu={mu}, sigma^2={sigma2})")

    # Case 1: Small variance on both X and Y
    x1 = np.random.normal(mu, np.sqrt(0.1), n_points)
    x2 = np.random.normal(mu, np.sqrt(0.1), n_points)
    y = a * x1 + b * x2 + c + np.random.normal(mu, np.sqrt(0.1), n_points)
    X = np.column_stack((np.ones(n_points), x1, x2))
    leverage_scores = compute_leverage_scores(X)
    ax = fig.add_subplot(2, 2, 1, projection='3d')
    scatter = ax.scatter(x1, x2, y, c=leverage_scores, cmap="viridis", s=20)
    ax.set_title("Small variance on both X and Y")
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_zlabel("Y")
    fig.colorbar(scatter, ax=ax, label="Leverage Score")

    # Case 2: High variance on X, small variance on Y
    x1 = np.random.normal(mu, np.sqrt(sigma2), n_points)
    x2 = np.random.normal(mu, np.sqrt(0.1), n_points)
    y = a * x1 + b * x2 + c + np.random.normal(mu, np.sqrt(0.1), n_points)
    X = np.column_stack((np.ones(n_points), x1, x2))
    leverage_scores = compute_leverage_scores(X)
    ax = fig.add_subplot(2, 2, 2, projection='3d')
    scatter = ax.scatter(x1, x2, y, c=leverage_scores, cmap="viridis", s=20)
    ax.set_title("High variance on X, small variance on Y")
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_zlabel("Y")
    fig.colorbar(scatter, ax=ax, label="Leverage Score")

    # Case 3: Small variance on X, high variance on Y
    x1 = np.random.normal(mu, np.sqrt(0.1), n_points)
    x2 = np.random.normal(mu, np.sqrt(sigma2), n_points)
    y = a * x1 + b * x2 + c + np.random.normal(mu, np.sqrt(sigma2), n_points)
    X = np.column_stack((np.ones(n_points), x1, x2))
    leverage_scores = compute_leverage_scores(X)
    ax = fig.add_subplot(2, 2, 3, projection='3d')
    scatter = ax.scatter(x1, x2, y, c=leverage_scores, cmap="viridis", s=20)
    ax.set_title("Small variance on X, high variance on Y")
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_zlabel("Y")
    fig.colorbar(scatter, ax=ax, label="Leverage Score")

    # Case 4: High variance on both X and Y
    x1 = np.random.normal(mu, np.sqrt(sigma2), n_points)
    x2 = np.random.normal(mu, np.sqrt(sigma2), n_points)
    y = a * x1 + b * x2 + c + np.random.normal(mu, np.sqrt(sigma2), n_points)
    X = np.column_stack((np.ones(n_points), x1, x2))
    leverage_scores = compute_leverage_scores(X)
    ax = fig.add_subplot(2, 2, 4, projection='3d')
    scatter = ax.scatter(x1, x2, y, c=leverage_scores, cmap="viridis", s=20)
    ax.set_title("High variance on both X and Y")
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_zlabel("Y")
    fig.colorbar(scatter, ax=ax, label="Leverage Score")

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.show()
