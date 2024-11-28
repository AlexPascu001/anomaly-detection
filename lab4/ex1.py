from pyod.utils.data import generate_data
from pyod.models.ocsvm import OCSVM
from pyod.models.deep_svdd import DeepSVDD
import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

n_train = 300
n_test = 200
contamination = 0.15
n_features = 3

X_train, X_test, y_train, y_test = generate_data(n_train=n_train, n_test=n_test, n_features=n_features, contamination=contamination, random_state=112)

ocsvm = OCSVM(kernel='linear', contamination=0.3)
ocsvm.fit(X_train)

y_pred = ocsvm.predict(X_test)

balanced_acc = balanced_accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, ocsvm.decision_function(X_test))

print(f"Balanced Accuracy: {balanced_acc}")
print(f"ROC AUC: {roc_auc}")

def plot_3d_data(ax, X, labels, title):
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels, cmap='viridis', marker='o')
    ax.set_title(title)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_zlabel("Feature 3")

fig = plt.figure(figsize=(16, 12))

# Ground Truth for Training Data
ax1 = fig.add_subplot(221, projection='3d')
plot_3d_data(ax1, X_train, y_train, "Ground Truth (Train)")

# Ground Truth for Test Data
ax2 = fig.add_subplot(222, projection='3d')
plot_3d_data(ax2, X_test, y_test, "Ground Truth (Test)")

# Predicted Labels for Training Data
ax3 = fig.add_subplot(223, projection='3d')
plot_3d_data(ax3, X_train, ocsvm.labels_, "Predicted Labels (Train)")

# Predicted Labels for Test Data
ax4 = fig.add_subplot(224, projection='3d')
plot_3d_data(ax4, X_test, y_pred, "Predicted Labels (Test)")

plt.tight_layout()
plt.show()

# Train OCSVM model with RBF kernel
ocsvm_rbf = OCSVM(kernel='rbf', contamination=0.3)
ocsvm_rbf.fit(X_train)

y_pred_train_rbf = ocsvm_rbf.predict(X_train)
y_pred_test_rbf = ocsvm_rbf.predict(X_test)

balanced_acc_rbf = balanced_accuracy_score(y_test, y_pred_test_rbf)
roc_auc_rbf = roc_auc_score(y_test, ocsvm_rbf.decision_function(X_test))

print(f"Balanced Accuracy (RBF): {balanced_acc_rbf}")
print(f"ROC AUC (RBF): {roc_auc_rbf}")


fig = plt.figure(figsize=(16, 12))

# Ground Truth for Training Data
ax1 = fig.add_subplot(221, projection='3d')
plot_3d_data(ax1, X_train, y_train, "Ground Truth (Train)")

# Ground Truth for Test Data
ax2 = fig.add_subplot(222, projection='3d')
plot_3d_data(ax2, X_test, y_test, "Ground Truth (Test)")
# Predicted Labels for Training Data
ax3 = fig.add_subplot(223, projection='3d')
plot_3d_data(ax3, X_train, y_pred_train_rbf, "Predicted Labels (Train) - RBF")

# Predicted Labels for Test Data
ax4 = fig.add_subplot(224, projection='3d')
plot_3d_data(ax4, X_test, y_pred_test_rbf, "Predicted Labels (Test) - RBF")

plt.tight_layout()
plt.show()

deep_svdd = DeepSVDD(contamination=0.3, n_features=n_features)
deep_svdd.fit(X_train)

y_pred_deepsvdd = deep_svdd.predict(X_test)

balanced_acc_deepsvdd = balanced_accuracy_score(y_test, y_pred_deepsvdd)
roc_auc_deepsvdd = roc_auc_score(y_test, deep_svdd.decision_function(X_test))

print(f"Balanced Accuracy (DeepSVDD): {balanced_acc_deepsvdd}")
print(f"ROC AUC (DeepSVDD): {roc_auc_deepsvdd}")

fig = plt.figure(figsize=(16, 12))

# Ground Truth for Training Data
ax1 = fig.add_subplot(221, projection='3d')
plot_3d_data(ax1, X_train, y_train, "Ground Truth (Train)")

# Ground Truth for Test Data
ax2 = fig.add_subplot(222, projection='3d')
plot_3d_data(ax2, X_test, y_test, "Ground Truth (Test)")
# Predicted Labels for Training Data
ax3 = fig.add_subplot(223, projection='3d')
plot_3d_data(ax3, X_train, deep_svdd.labels_, "Predicted Labels (Train) - DeepSVDD")

# Predicted Labels for Test Data
ax4 = fig.add_subplot(224, projection='3d')
plot_3d_data(ax4, X_test, y_pred_deepsvdd, "Predicted Labels (Test) - DeepSVDD")

plt.tight_layout()
plt.show()