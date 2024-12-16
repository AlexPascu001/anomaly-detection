import numpy as np
from pyod.models.pca import PCA
from pyod.models.kpca import KPCA
from sklearn.model_selection import train_test_split
from pyod.utils.utility import standardizer
from sklearn.metrics import balanced_accuracy_score
from scipy.io import loadmat
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('TkAgg')

data = loadmat("shuttle.mat")
X = data["X"]
y = data["y"].ravel()

n_features = X.shape[1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=112)

X_train_norm, X_test_norm = standardizer(X_train, X_test)

contamination_rate = 0.07 # from ODDS website

# Plot the cumulative explained variance
# and the individual variances as in the previous exercise (you can access
# the variances with the explained variance attribute)

pca = PCA(n_components=2, contamination=contamination_rate)
pca.fit(X_train_norm)

explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

# Plot cumulative and individual variances
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.6, label='Individual Variance')
plt.step(range(1, len(cumulative_variance) + 1), cumulative_variance, where='mid', label='Cumulative Variance', color='red')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.legend()
plt.title('Explained Variance - PCA')
plt.show()

# Balance accuracy score for both training and testing sets
train_bal_acc = balanced_accuracy_score(y_train, pca.predict(X_train_norm))
test_bal_acc = balanced_accuracy_score(y_test, pca.predict(X_test_norm))

print("======== PCA =========")
print(f"Train Balanced Accuracy: {train_bal_acc}")
print(f"Test Balanced Accuracy: {test_bal_acc}")

# Repeat the same process for KPCA
kpca = KPCA(n_components=2, contamination=contamination_rate)
kpca.fit(X_train_norm)

explained_variance = kpca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

# Plot cumulative and individual variances
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.6, label='Individual Variance')
plt.step(range(1, len(cumulative_variance) + 1), cumulative_variance, where='mid', label='Cumulative Variance', color='red')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.legend()
plt.title('Explained Variance - KPCA')
plt.show()

# Balance accuracy score for both training and testing sets
train_bal_acc = balanced_accuracy_score(y_train, kpca.predict(X_train_norm))
test_bal_acc = balanced_accuracy_score(y_test, kpca.predict(X_test_norm))

print("======== KPCA =========")
print(f"Train Balanced Accuracy: {train_bal_acc}")
print(f"Test Balanced Accuracy: {test_bal_acc}")