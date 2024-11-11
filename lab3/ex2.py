from sklearn.datasets import make_blobs
import numpy as np
from pyod.models.iforest import IForest
import matplotlib.pyplot as plt
from pyod.models.dif import DIF
from pyod.models.loda import LODA

import matplotlib
matplotlib.use('TkAgg')

centers = [(10, 0), (0, 10)]
n_samples = 500
X_train, _ = make_blobs(n_samples=[n_samples, n_samples], centers=centers, cluster_std=1.0, random_state=42)
X_train = np.concatenate((X_train, np.array([(-10, -10), (-10, 20), (20, -10), (20, 20)])), axis=0)

contamination_rate = 0.02
n_neurons = [500, 200]
n_bins = 50

iforest = IForest(contamination=contamination_rate, random_state=112)
iforest.fit(X_train)

X_test = np.random.uniform(-10, 20, (1000, 2))
iforest_scores = iforest.decision_function(X_test)

dif = DIF(contamination=contamination_rate, random_state=112, hidden_neurons=n_neurons)
dif.fit(X_train)
dif_scores = dif.decision_function(X_test)

loda = LODA(contamination=contamination_rate, n_bins=n_bins)
loda.fit(X_train)
loda_scores = loda.decision_function(X_test)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].scatter(X_test[:, 0], X_test[:, 1], c=iforest_scores, cmap='plasma')
axes[0].set_title("Isolation Forest")
axes[1].scatter(X_test[:, 0], X_test[:, 1], c=dif_scores, cmap='plasma')
axes[1].set_title("Deep Isolation Forest")
axes[2].scatter(X_test[:, 0], X_test[:, 1], c=loda_scores, cmap='plasma')
axes[2].set_title("LODA")
for ax in axes:
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
fig.colorbar(axes[2].collections[0], ax=axes, orientation='horizontal', label='Anomaly Score')
plt.show()

centers_3d = [(0, 10, 0), (10, 0, 10)]
X_train_3d, _ = make_blobs(n_samples=[n_samples, n_samples], centers=centers_3d, cluster_std=1.0, random_state=42)
X_train_3d = np.concatenate((X_train_3d, np.array([(-10, -10, -10), (-10, 20, 20), (20, -10, 20), (20, 20, -10)])), axis=0)

iforest_3d = IForest(contamination=contamination_rate, random_state=112)
iforest_3d.fit(X_train_3d)
X_test_3d = np.random.uniform(-10, 20, (1000, 3))
test_scores_iforest_3d = iforest_3d.decision_function(X_test_3d)

dif_3d = DIF(contamination=contamination_rate, random_state=112, hidden_neurons=n_neurons)
dif_3d.fit(X_train_3d)
test_scores_dif_3d = dif_3d.decision_function(X_test_3d)

loda_3d = LODA(contamination=contamination_rate, n_bins=n_bins)
loda_3d.fit(X_train_3d)
test_scores_loda_3d = loda_3d.decision_function(X_test_3d)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
p = ax.scatter(X_test_3d[:, 0], X_test_3d[:, 1], X_test_3d[:, 2], c=test_scores_iforest_3d, cmap='plasma')
fig.colorbar(p, ax=ax, label='Anomaly Score (Isolation Forest)')
ax.set_title('3D Isolation Forest Anomaly Scores')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('X3')
plt.show()

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
p = ax.scatter(X_test_3d[:, 0], X_test_3d[:, 1], X_test_3d[:, 2], c=test_scores_dif_3d, cmap='plasma')
fig.colorbar(p, ax=ax, label='Anomaly Score (Deep Isolation Forest)')
ax.set_title('3D Deep Isolation Forest Anomaly Scores')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('X3')
plt.show()

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
p = ax.scatter(X_test_3d[:, 0], X_test_3d[:, 1], X_test_3d[:, 2], c=test_scores_loda_3d, cmap='plasma')
fig.colorbar(p, ax=ax, label='Anomaly Score (LODA)')
ax.set_title('3D LODA Anomaly Scores')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('X3')
plt.show()