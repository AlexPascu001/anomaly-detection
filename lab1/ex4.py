import numpy as np
from sklearn.metrics import confusion_matrix, balanced_accuracy_score

mean = np.zeros(2)
cov = np.eye(2)
n_samples = 1000
data = np.random.multivariate_normal(mean, cov, n_samples)

new_cov = np.array([[1.0, 0.6], [0.6, 2.0]])
mean_shifted = np.array([0.0, 0.0])
anomalous_samples = int(n_samples * 0.1)
shifted_data = np.random.multivariate_normal(mean_shifted, new_cov, anomalous_samples)

combined_data = np.vstack([data, shifted_data])

L = np.linalg.cholesky(new_cov)

def z_score(x, mean, L):
    y = x - mean
    f = np.linalg.solve(L, y)
    distance = np.dot(f.T, f)
    return np.sqrt(distance)

z_scores = np.array([z_score(x, mean_shifted, L) for x in combined_data])

contamination = 0.1
threshold = np.quantile(z_scores, 1 - contamination)

predicted_anomalies = (z_scores > threshold).astype(int)

true_labels = np.zeros(len(combined_data))
true_labels[-anomalous_samples:] = 1  # Last 10% are anomalies

cm = confusion_matrix(true_labels, predicted_anomalies)
tn, fp, fn, tp = cm.ravel()
tnr = tn / (tn + fp)
tpr = tp / (tp + fn)
ba = (tpr + tnr) / 2
print(f"Confusion Matrix:\n{cm}")
print(f"True Negative Rate (TNR): {tnr:.2f}")
print(f"True Positive Rate (TPR): {tpr:.2f}")
print(f"Balanced Accuracy: {ba:.2f}")
