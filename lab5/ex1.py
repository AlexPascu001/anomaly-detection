import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eigh

import matplotlib
matplotlib.use('TkAgg')


mean = [5, 10, 2]
cov = [[3, 2, 2], [2, 10, 1], [2, 1, 2]]
data = np.random.multivariate_normal(mean, cov, 500)

# Center the data
data -= np.mean(data, axis=0)

# Calculate the covariance matrix
cov_matrix = np.cov(data, rowvar=False)

# Calculate the eigenvalues and eigenvectors
eigenvalues, eigenvectors = eigh(cov_matrix)

# Sort the eigenvectors by decreasing eigenvalues
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

# Project the data onto the eigenvectors
projected_data = np.dot(data, eigenvectors)

# Plot the original data
fig = plt.figure(figsize=(16, 12))
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(data[:, 0], data[:, 1], data[:, 2], c='b', marker='o')
ax1.set_title("Original Data")
ax1.set_xlabel("Feature 1")
ax1.set_ylabel("Feature 2")
ax1.set_zlabel("Feature 3")

# Plot the projected data
ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(projected_data[:, 0], projected_data[:, 1], projected_data[:, 2], c='r', marker='o')
ax2.set_title("Projected Data")
ax2.set_xlabel("Principal Component 1")
ax2.set_ylabel("Principal Component 2")
ax2.set_zlabel("Principal Component 3")

plt.tight_layout()
plt.show()

# Compute cumulative explained variance
cumulative_variance = np.cumsum(eigenvalues) / np.sum(eigenvalues)
individual_variance = eigenvalues / np.sum(eigenvalues)

# Plot cumulative and individual variances
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(individual_variance) + 1), individual_variance, alpha=0.6, label='Individual Variance')
plt.step(range(1, len(cumulative_variance) + 1), cumulative_variance, where='mid', label='Cumulative Variance')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.legend()
plt.title('Explained Variance')
plt.show()

# Focus on the 3rd principal component
pc3 = projected_data[:, 2]

# Compute thresholds for outliers
contamination_rate = 0.1
threshold_pc3 = np.quantile(pc3, [contamination_rate, 1 - contamination_rate])

# Label outliers
labels_pc3 = (pc3 < threshold_pc3[0]) | (pc3 > threshold_pc3[1])

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(data[:, 0], data[:, 1], c=~labels_pc3, cmap='coolwarm', alpha=0.7)
plt.title('Outliers Detected Using 3rd Principal Component')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Focus on the 2nd principal component
pc2 = projected_data[:, 1]

# Compute thresholds for outliers
threshold_pc2 = np.quantile(pc2, [contamination_rate, 1 - contamination_rate])

# Label outliers
labels_pc2 = (pc2 < threshold_pc2[0]) | (pc2 > threshold_pc2[1])

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(data[:, 0], data[:, 1], c=~labels_pc2, cmap='coolwarm', alpha=0.7)
plt.title('Outliers Detected Using 2nd Principal Component')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Normalize the data in the PCA space
normalized_data = projected_data / np.std(projected_data, axis=0)

# Compute squared Euclidean distance to the centroid
squared_distances = np.sum(normalized_data**2, axis=1)

# Define a threshold for outliers
threshold_distance = np.quantile(squared_distances, 1 - contamination_rate)

# Label outliers
labels_distance = squared_distances > threshold_distance

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(data[:, 0], data[:, 1], c=~labels_distance, cmap='viridis', alpha=0.7)
plt.title('Outliers Detected Using Normalized Distance')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
