from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pyod.models.lof import LOF
from sklearn.metrics import balanced_accuracy_score
from pyod.models.combination import average, maximization
import numpy as np

data = loadmat('cardio.mat')
X = data['X']
y = data['y'].ravel()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=112)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

train_scores = []
test_scores = []
neighbors_range = range(100, 301, 10)

for n_neighbors in neighbors_range:
    lof = LOF(n_neighbors=n_neighbors, novelty=True)
    lof.fit(X_train)

    train_scores.append(lof.decision_function(X_train))
    test_scores.append(lof.decision_function(X_test))

train_scores = np.array(train_scores).T
test_scores = np.array(test_scores).T

train_scores = StandardScaler().fit_transform(train_scores)
test_scores = StandardScaler().fit_transform(test_scores)

combined_scores = np.concatenate([train_scores, test_scores], axis=0)

avg_scores = average(combined_scores)
max_scores = maximization(combined_scores)

contamination_rate = 0.096  # from dataset description
threshold_avg = np.quantile(avg_scores[-len(y_test):], 1 - contamination_rate)
threshold_max = np.quantile(max_scores[-len(y_test):], 1 - contamination_rate)

avg_pred = (avg_scores[-len(y_test):] >= threshold_avg).astype(int)
max_pred = (max_scores[-len(y_test):] >= threshold_max).astype(int)

ba_avg = balanced_accuracy_score(y_test, avg_pred)
ba_max = balanced_accuracy_score(y_test, max_pred)

print("Balanced Accuracy with Average strategy:", ba_avg)
print("Balanced Accuracy with Maximization strategy:", ba_max)
