from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('TkAgg')

n_samples = 500
X, _ = make_blobs(n_samples=n_samples, centers=1, center_box=(0, 0), cluster_std=1.0, random_state=112)

num_projections = 5
projections = []

for _ in range(num_projections):
    vector = np.random.multivariate_normal([0, 0], np.identity(2))
    unit_vector = vector / np.linalg.norm(vector)
    projections.append(unit_vector)

histograms = []
bins = 25
range_min, range_max = -10, 10

for vector in projections:
    projected_data = X @ vector
    hist, bin_edges = np.histogram(projected_data, bins=bins, range=(range_min, range_max), density=True)
    histograms.append((hist, bin_edges))

def compute_anomaly_score(point, projections, histograms):
    scores = []
    for vector, (hist, bin_edges) in zip(projections, histograms):
        projected_value = point @ vector
        bin_index = np.digitize(projected_value, bin_edges) - 1
        bin_index = min(max(bin_index, 0), len(hist) - 1)
        probability = hist[bin_index]
        scores.append(probability)
    return np.mean(scores)

anomaly_scores = np.array([compute_anomaly_score(x, projections, histograms) for x in X])

test_data = np.random.uniform(-3, 3, (1000, 2))

test_scores = np.array([compute_anomaly_score(x, projections, histograms) for x in test_data])
train_scores = np.array([compute_anomaly_score(x, projections, histograms) for x in X])

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

sc_train = axes[0].scatter(X[:, 0], X[:, 1], c=train_scores, cmap='plasma')
fig.colorbar(sc_train, ax=axes[0], label='Anomaly Score')
axes[0].set_xlabel('X1')
axes[0].set_ylabel('X2')
axes[0].set_title('Anomaly Scores for Train Dataset')

sc_test = axes[1].scatter(test_data[:, 0], test_data[:, 1], c=test_scores, cmap='plasma')
fig.colorbar(sc_test, ax=axes[1], label='Anomaly Score')
axes[1].set_xlabel('X1')
axes[1].set_ylabel('X2')
axes[1].set_title('Anomaly Scores for Test Dataset')

plt.show()