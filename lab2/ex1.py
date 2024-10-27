import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from numpy.linalg import inv

a, b, c = 2, 3, 5
n_points = 100
mu_values = [0, 0.5, 1]
sigma2_values = [0.1, 1, 2]


def compute_leverage_scores(X):
    H = X @ inv(X.T @ X) @ X.T
    leverage_scores = np.diag(H)
    return leverage_scores


def plot_with_leverage_scores(X, y, leverage_scores, title, ax):
    ax.scatter(X[:, 1], y, color="blue", label="Data Points")
    high_leverage_points = leverage_scores > np.percentile(leverage_scores, 90)
    ax.scatter(X[high_leverage_points, 1], y[high_leverage_points], color="red", label="High Leverage")
    ax.set_title(title)
    ax.legend()


fig, axes = plt.subplots(len(mu_values), len(sigma2_values), figsize=(15, 10))
fig.suptitle("Leverage Scores with Different Noise Variances and Means")

for i, mu in enumerate(mu_values):
    for j, sigma2 in enumerate(sigma2_values):
        x1 = np.random.uniform(-10, 10, n_points)

        regular_x = x1
        high_var_x = x1 * np.random.normal(1, np.sqrt(sigma2), n_points)
        high_var_y = a * x1 + b + np.random.normal(mu, np.sqrt(sigma2), n_points)
        high_var_xy = high_var_x * np.random.normal(1, np.sqrt(sigma2), n_points)

        y = a * x1 + b + np.random.normal(mu, np.sqrt(sigma2), n_points)

        X = np.column_stack((np.ones(n_points), x1))
        leverage_scores = compute_leverage_scores(X)

        plot_with_leverage_scores(X, y, leverage_scores, f"mu={mu}, sigma^2={sigma2}", axes[i, j])

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.show()

fig, axes = plt.subplots(len(mu_values), len(sigma2_values), figsize=(15, 10))
fig.suptitle("Leverage Scores in 2D Case")

for i, mu in enumerate(mu_values):
    for j, sigma2 in enumerate(sigma2_values):
        x1 = np.random.uniform(-10, 10, n_points)
        x2 = np.random.uniform(-10, 10, n_points)

        y = a * x1 + b * x2 + c + np.random.normal(mu, np.sqrt(sigma2), n_points)

        X = np.column_stack((np.ones(n_points), x1, x2))
        leverage_scores = compute_leverage_scores(X)

        scatter = axes[i, j].scatter(x1, x2, c=leverage_scores, cmap="viridis")
        axes[i, j].set_title(f"mu={mu}, sigma^2={sigma2}")
        fig.colorbar(scatter, ax=axes[i, j], label="Leverage Score")

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.show()
