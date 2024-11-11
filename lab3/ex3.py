from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from pyod.models.iforest import IForest
from pyod.models.dif import DIF
from pyod.models.loda import LODA
import matplotlib.pyplot as plt
from time import time

import matplotlib
matplotlib.use('TkAgg')

data = loadmat('shuttle.mat')
X = data['X']
y = data['y'].ravel()

contamination_rate = 0.07 # from ODDS website
# Multi-dimensional point datasets ; Shuttle, 49097, 9, 3511 (7%).

tries = 10
if_balanced_accuracies, if_roc_aucs = [], []
dif_balanced_accuracies, dif_roc_aucs = [], []
loda_balanced_accuracies, loda_roc_aucs = [], []

for i in range(tries):
    start = time()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    end = time()
    print("Time to split and scale:", end - start)

    # IForest
    start = time()
    iforest = IForest(n_estimators=100, contamination=contamination_rate)
    iforest.fit(X_train)
    iforest_pred = iforest.predict(X_test)
    if_balanced_accuracies.append(balanced_accuracy_score(y_test, iforest_pred))
    if_roc_aucs.append(roc_auc_score(y_test, iforest_pred))
    end = time()
    print("Time to fit and predict IForest:", end - start)

    # DIF
    start = time()
    dif = DIF(contamination=contamination_rate)
    dif.fit(X_train)
    dif_pred = dif.predict(X_test)
    dif_balanced_accuracies.append(balanced_accuracy_score(y_test, dif_pred))
    dif_roc_aucs.append(roc_auc_score(y_test, dif_pred))
    end = time()
    print("Time to fit and predict DIF:", end - start)

    # LODA
    start = time()
    loda = LODA(contamination=contamination_rate)
    loda.fit(X_train)
    loda_pred = loda.predict(X_test)
    loda_balanced_accuracies.append(balanced_accuracy_score(y_test, loda_pred))
    loda_roc_aucs.append(roc_auc_score(y_test, loda_pred))
    end = time()
    print("Time to fit and predict LODA:", end - start)

# Mean BA
print("IForest Balanced Accuracy:", np.mean(if_balanced_accuracies))
print("DIF Balanced Accuracy:", np.mean(dif_balanced_accuracies))
print("LODA Balanced Accuracy:", np.mean(loda_balanced_accuracies))

# Mean ROC AUC
print("IForest ROC AUC:", np.mean(if_roc_aucs))
print("DIF ROC AUC:", np.mean(dif_roc_aucs))
print("LODA ROC AUC:", np.mean(loda_roc_aucs))
