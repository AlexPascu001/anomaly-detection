from scipy.io import loadmat
from pyod.utils.utility import standardizer
from pyod.models.ocsvm import OCSVM
from pyod.models.deep_svdd import DeepSVDD
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

data = loadmat("shuttle.mat")
X = data["X"]
y = data["y"].ravel()

print(X.shape, y.shape)

n_features = X.shape[1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=112)

X_train_norm, X_test_norm = standardizer(X_train, X_test)
contamination_rate = 0.07 # from ODDS website

# OCSVM
ocsvm = OCSVM(contamination=contamination_rate)
ocsvm.fit(X_train_norm)

y_pred = ocsvm.predict(X_test_norm)
y_scores = ocsvm.decision_function(X_test_norm)
ocsvm_roc = roc_auc_score(y_test, y_scores)
ocsvm_bal_acc = balanced_accuracy_score(y_test, y_pred)

# Deep SVDD
deep_svdd = DeepSVDD(contamination=contamination_rate, n_features=n_features)
deep_svdd.fit(X_train_norm)

y_pred = deep_svdd.predict(X_test_norm)
y_scores = deep_svdd.decision_function(X_test_norm)
svdd_roc = roc_auc_score(y_test, y_scores)
svdd_bal_acc = balanced_accuracy_score(y_test, y_pred)

# Other architectures for Deep SVDD
hidden_neurons_list = [
    [32, 16, 8],
    [64, 32, 16],
    [128, 64, 32],
    [32, 16, 8, 16, 32],
    [32, 16, 8, 4, 2]
]

rocs, bal_accs = [], []
best_roc, best_bal_acc = 0, 0
best_hidden_neurons = None

for hidden_neurons in hidden_neurons_list:
    print(f"Hidden Neurons: {hidden_neurons}")
    deep_svdd = DeepSVDD(contamination=contamination_rate, n_features=n_features, hidden_neurons=hidden_neurons)
    deep_svdd.fit(X_train_norm)

    y_pred = deep_svdd.predict(X_test_norm)
    y_scores = deep_svdd.decision_function(X_test_norm)
    roc = roc_auc_score(y_test, y_scores)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    rocs.append(roc)
    bal_accs.append(bal_acc)

    if roc > best_roc:
        best_roc = roc
        best_bal_acc = bal_acc
        best_hidden_neurons = hidden_neurons

print("=====OCSVM=====")
print(f"ROC: {ocsvm_roc}, Balanced Accuracy: {ocsvm_bal_acc}")
print("=====Deep SVDD=====")
print(f"ROC: {svdd_roc}, Balanced Accuracy: {svdd_bal_acc}")
print("=====Deep SVDD with different architectures=====")
for i, (roc, bal_acc) in enumerate(zip(rocs, bal_accs)):
    print(f"Hidden Neurons: {hidden_neurons_list[i]}, ROC: {roc}, Balanced Accuracy: {bal_acc}")
print(f"Best Hidden Neurons: {best_hidden_neurons}, ROC: {best_roc}, Balanced Accuracy: {best_bal_acc}")