from sklearn.datasets import make_blobs
from sklearn.utils import shuffle
from pyod.models.knn import KNN
from pyod.models.lof import LOF
import numpy as np
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('TkAgg')

n_samples_1 = 200
n_samples_2 = 100
center_1 = (-10, -10)
center_2 = (10, 10)
std_1 = 2
std_2 = 6

X_cluster_1, y_cluster_1 = make_blobs(n_samples=n_samples_1, n_features=2, center_box=center_1, cluster_std=std_1, random_state=112)
X_cluster_2, y_cluster_2 = make_blobs(n_samples=n_samples_2, n_features=2, center_box=center_2, cluster_std=std_2, random_state=112)

X_train = np.concatenate([X_cluster_1, X_cluster_2], axis=0)
y_train = np.concatenate([y_cluster_1, y_cluster_2], axis=0)
X_train, y_train = shuffle(X_train, y_train)

contamination = 0.07
n_neighbors = 7

clf_knn = KNN(contamination=contamination, n_neighbors=n_neighbors)
clf_knn.fit(X_train)

clf_lof = LOF(contamination=contamination, n_neighbors=n_neighbors)
clf_lof.fit(X_train)

fig, axes = plt.subplots(1, 2, figsize=(14, 7))
fig.suptitle(f"Outlier Detection with KNN and LOF (n_neighbors={n_neighbors})")

# KNN Predictions
pred_knn = clf_knn.predict(X_train)
for i in range(len(X_train)):
    if pred_knn[i] == 0:
        axes[0].scatter(X_train[i, 0], X_train[i, 1], color='blue', alpha=0.8)
    else:
        axes[0].scatter(X_train[i, 0], X_train[i, 1], color='red', alpha=0.8)
axes[0].set_title("Predicted Labels for KNN")

# LOF Predictions
pred_lof = clf_lof.predict(X_train)
for i in range(len(X_train)):
    if pred_lof[i] == 0:
        axes[1].scatter(X_train[i, 0], X_train[i, 1], color='blue', alpha=0.8)
    else:
        axes[1].scatter(X_train[i, 0], X_train[i, 1], color='red', alpha=0.8)
axes[1].set_title("Predicted Labels for LOF")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
