import matplotlib.pyplot as plt
import matplotlib
from pyod.utils.data import generate_data

matplotlib.use('TkAgg')


#%% Ex 1

n_train = 400
n_test = 100
contamination = 0.1
n_features = 2

X_train, X_test, y_train, y_test = generate_data(n_train=n_train, n_test=n_test, n_features=n_features, contamination=contamination)

for i in range(n_train):
    if y_train[i] == 0:
        plt.scatter(X_train[i, 0], X_train[i, 1], c='blue', alpha=0.8)
    else:
        plt.scatter(X_train[i, 0], X_train[i, 1], c='red', alpha=0.8)
plt.show()
