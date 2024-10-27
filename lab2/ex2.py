from pyod.utils.data import generate_data_clusters
from pyod.models.knn import KNN
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

n_train = 400
n_test = 200
n_clusters = 2
n_features = 2
contamination = 0.1

X_train, X_test, y_train, y_test = generate_data_clusters(n_train=n_train, n_test=n_test, n_clusters=n_clusters, n_features=n_features, contamination=contamination, random_state=112)

clf = KNN(contamination=contamination, n_neighbors=5)
clf.fit(X_train)

fig, axes = plt.subplots(2, 2, figsize=(10, 10))

fig.suptitle(f"KNN n_neighbors={clf.n_neighbors}")

for i in range(n_train):
    if y_train[i] == 0:
        axes[0, 0].scatter(X_train[i, 0], X_train[i, 1], c='blue', alpha=0.8)
    else:
        axes[0, 0].scatter(X_train[i, 0], X_train[i, 1], c='red', alpha=0.8)
axes[0, 0].set_title("Ground Truth Labels for Training Data")

y_train_pred = clf.predict(X_train)
for i in range(n_train):
    if y_train_pred[i] == 0:
        axes[0, 1].scatter(X_train[i, 0], X_train[i, 1], c='blue', alpha=0.8)
    else:
        axes[0, 1].scatter(X_train[i, 0], X_train[i, 1], c='red', alpha=0.8)
axes[0, 1].set_title("Predicted Labels for Training Data")

for i in range(n_test):
    if y_test[i] == 0:
        axes[1, 0].scatter(X_test[i, 0], X_test[i, 1], c='blue', alpha=0.8)
    else:
        axes[1, 0].scatter(X_test[i, 0], X_test[i, 1], c='red', alpha=0.8)
axes[1, 0].set_title("Ground Truth Labels for Test Data")

y_test_pred = clf.predict(X_test)
for i in range(n_test):
    if y_test_pred[i] == 0:
        axes[1, 1].scatter(X_test[i, 0], X_test[i, 1], c='blue', alpha=0.8)
    else:
        axes[1, 1].scatter(X_test[i, 0], X_test[i, 1], c='red', alpha=0.8)
axes[1, 1].set_title("Predicted Labels for Test Data")

plt.tight_layout()
plt.show()

cm = confusion_matrix(y_train, y_train_pred)
tn, fp, fn, tp = cm.ravel()
tpr = tp / (tp + fn)
tnr = tn / (tn + fp)
ba = (tpr + tnr) / 2
print(f"Balanced Accuracy for Training Data: {ba:.2f}")

cm = confusion_matrix(y_test, y_test_pred)
tn, fp, fn, tp = cm.ravel()
tpr = tp / (tp + fn)
tnr = tn / (tn + fp)
ba = (tpr + tnr) / 2
print(f"Balanced Accuracy for Test Data: {ba:.2f}")