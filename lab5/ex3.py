import numpy as np
from pyod.models.pca import PCA
from pyod.models.kpca import KPCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import keras
from keras.models import Model
from keras.layers import Input, Dense
from keras import Sequential
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
from keras.activations import relu, sigmoid

import matplotlib
matplotlib.use('TkAgg')

data = loadmat("shuttle.mat")
X = data["X"]
y = data["y"].ravel()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=112)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

class Autoencoder(Model):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Sequential([
            Dense(8, activation=relu),
            Dense(5, activation=relu),
            Dense(3, activation=relu),
        ])
        self.decoder = Sequential([
            Dense(5, activation=relu),
            Dense(8, activation=relu),
            Dense(9, activation=sigmoid),
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


autoencoder = Autoencoder()
autoencoder.compile(optimizer=Adam(), loss='mse')

history = autoencoder.fit(
    X_train, X_train,
    epochs=100,
    batch_size=1024,
    validation_data=(X_test, X_test),
)

plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

reconstructed_train = autoencoder.predict(X_train)
train_errors = np.mean(np.square(X_train - reconstructed_train), axis=1)

contamination_rate = 0.07
threshold = np.quantile(train_errors, 1 - contamination_rate)

train_labels = (train_errors > threshold).astype(int)

reconstructed_test = autoencoder.predict(X_test)
test_errors = np.mean(np.square(X_test - reconstructed_test), axis=1)
test_labels = (test_errors > threshold).astype(int)

train_bal_acc = balanced_accuracy_score(y_train, train_labels)
test_bal_acc = balanced_accuracy_score(y_test, test_labels)

print(f"Threshold: {threshold:.4f}")
print("======== Autoencoder Results ========")
print(f"Train Balanced Accuracy: {train_bal_acc:.4f}")
print(f"Test Balanced Accuracy: {test_bal_acc:.4f}")