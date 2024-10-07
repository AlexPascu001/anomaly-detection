import numpy as np
from pyod.utils.data import generate_data
from sklearn.metrics import confusion_matrix
#%% Ex 3
n_train = 1000
n_test = 0
contamination = 0.1
n_features = 1

X_train, X_test, y_train, y_test = generate_data(n_train=n_train, n_test=n_test, n_features=n_features, contamination=contamination)
# detect anomalies in the training data using Z-Score
mean, std = np.mean(X_train), np.std(X_train)

z_scores = (X_train - mean) / std
threshold = np.quantile(np.abs(z_scores), 1 - contamination)
y_train_pred = (np.abs(z_scores) > threshold).astype(int)
cm = confusion_matrix(y_train, y_train_pred)
# compute balanced accuracy
tn, fp, fn, tp = cm.ravel()
tpr = tp / (tp + fn)
tnr = tn / (tn + fp)
ba = (tpr + tnr) / 2
print(f"Confusion Matrix:\n{cm}")
print(f"True Negative Rate (TNR): {tnr:.2f}")
print(f"True Positive Rate (TPR): {tpr:.2f}")
print(f"Balanced Accuracy: {ba:.2f}")

