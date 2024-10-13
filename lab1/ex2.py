#%% Ex 2
import sklearn
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')
from pyod.models.knn import KNN
from pyod.utils.data import generate_data
from sklearn.metrics import confusion_matrix, roc_curve

n_train = 400
n_test = 100
contamination = 0.5
n_features = 2

X_train, X_test, y_train, y_test = generate_data(n_train=n_train, n_test=n_test, n_features=n_features, contamination=contamination)


clf = KNN(contamination=contamination)
clf.fit(X_train)

print("For training data:")
y_train_pred = clf.predict(X_train)
y_train_scores = clf.decision_function(X_train)
y_test_pred = clf.predict(X_test)
y_test_scores = clf.decision_function(X_test)

cm_1 = confusion_matrix(y_train, y_train_pred)
print(cm_1)

# get TN, FP, FN, TP
tn, fp, fn, tp = cm_1.ravel()
print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")
tpr = tp / (tp + fn)
tnr = tn / (tn + fp)
ba = (tpr + tnr) / 2
print(f"TPR: {tpr}, TNR: {tnr}, BA: {ba}")

roc = roc_curve(y_train, y_train_scores)
plt.plot(roc[0], roc[1])
plt.show()
auc = sklearn.metrics.auc(roc[0], roc[1])
print(f"AUC: {auc}")

print("For test data:")
cm_2 = confusion_matrix(y_test, y_test_pred)
print(cm_2)
# get TN, FP, FN, TP
tn, fp, fn, tp = cm_1.ravel()
print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")
tpr = tp / (tp + fn)
tnr = tn / (tn + fp)
ba = (tpr + tnr) / 2
print(f"TPR: {tpr}, TNR: {tnr}, BA: {ba}")

roc = roc_curve(y_test, y_test_scores)
plt.plot(roc[0], roc[1])
plt.show()
auc = sklearn.metrics.auc(roc[0], roc[1])
print(f"AUC: {auc}")
