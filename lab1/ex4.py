import numpy as np
from sklearn.metrics import confusion_matrix, balanced_accuracy_score

mean = np.zeros(2)
cov = np.eye(2)
n_samples = 1000
data = np.random.multivariate_normal(mean, cov, n_samples)

new_cov = np.array([[1.0, 0.6], [0.6, 2.0]])
mean_shifted = np.array([0.0, 0.0])

shifted_data = np.random.multivariate_normal(mean_shifted, new_cov, n_samples)

L = np.linalg.cholesky(new_cov)

def z_score(x, mean, L):
    y = x - mean
    f = np.linalg.solve(L, y)
    distance = np.dot(f.T, f)
    return np.sqrt(distance)

z_scores = np.array([z_score(x, mean_shifted, L) for x in shifted_data])

contamination = 0.1
threshold = np.quantile(z_scores, 1 - contamination)

predicted_anomalies = (z_scores > threshold).astype(int)

true_labels = np.ones(len(shifted_data))
predicted_labels = predicted_anomalies

true_labels = np.concatenate([np.zeros(len(data)), true_labels])

predicted_labels_normal = np.zeros(len(data))
predicted_labels = np.concatenate([predicted_labels_normal, predicted_labels])

cm = confusion_matrix(true_labels, predicted_labels)
tn, fp, fn, tp = cm.ravel()
tnr = tn / (tn + fp)
tpr = tp / (tp + fn)
ba = (tpr + tnr) / 2
print(f"Confusion Matrix:\n{cm}")
print(f"True Negative Rate (TNR): {tnr:.2f}")
print(f"True Positive Rate (TPR): {tpr:.2f}")
print(f"Balanced Accuracy: {ba:.2f}")
