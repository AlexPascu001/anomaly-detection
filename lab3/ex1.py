from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('TkAgg')

n_samples = 500
X, _ = make_blobs(n_samples=n_samples, centers=1, cluster_std=1.0, random_state=112)

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

test_data = np.random.uniform(-3, 3, (500, 2))

test_scores = np.array([compute_anomaly_score(x, projections, histograms) for x in test_data])

plt.scatter(test_data[:, 0], test_data[:, 1], c=test_scores, cmap='plasma')
plt.colorbar(label='Anomaly Score')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Anomaly Scores for Test Dataset')
plt.show()