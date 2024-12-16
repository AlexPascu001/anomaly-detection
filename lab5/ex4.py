from tensorflow.keras.datasets import mnist
from tensorflow.random import normal
from tensorflow import clip_by_value
from keras import Sequential
from keras.layers import Conv2D, Conv2DTranspose, Flatten, Dense, Reshape
from keras.optimizers import Adam
from keras.activations import relu
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('TkAgg')

(X_train, _), (X_test, _) = mnist.load_data()

X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.

noise_factor = 0.35
X_train_noisy = X_train + noise_factor * normal(shape=X_train.shape)
X_test_noisy = X_test + noise_factor * normal(shape=X_test.shape)

# clip_by_value -> np.clip
# because tensors returned by clip_by_value
# don't support squeeze() method used for plotting
X_train_noisy = np.clip(X_train_noisy.numpy(), 0., 1.)
X_test_noisy = np.clip(X_test_noisy.numpy(), 0., 1.)

X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]
X_train_noisy = X_train_noisy[..., np.newaxis]
X_test_noisy = X_test_noisy[..., np.newaxis]

input_shape = X_train.shape[1:]


class ConvolutionalAutoencoder(Model):
    def __init__(self):
        super(ConvolutionalAutoencoder, self).__init__()
        self.encoder = Sequential([
            Conv2D(8, (3, 3), activation=relu, strides=2, padding='same'),
            Conv2D(4, (3, 3), activation=relu, strides=2, padding='same'),
        ])
        self.decoder = Sequential([
            Conv2DTranspose(4, (3, 3), activation=relu, strides=2, padding='same'),
            Conv2DTranspose(8, (3, 3), activation=relu, strides=2, padding='same'),
            Conv2D(1, (3, 3), activation='sigmoid', padding='same'),
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


autoencoder = ConvolutionalAutoencoder()
autoencoder.compile(optimizer=Adam(), loss='mse')

history = autoencoder.fit(
    X_train_noisy, X_train,
    epochs=10,
    batch_size=64,
    validation_data=(X_test_noisy, X_test)
)


def plot_images(original, noisy, reconstructed, n=5):
    plt.figure(figsize=(15, 6))
    for i in range(n):
        # Original images
        plt.subplot(4, n, i + 1)
        plt.imshow(original[i].squeeze(), cmap='gray')
        plt.axis('off')
        if i == 0: plt.title("Original")

        # Noisy images
        plt.subplot(4, n, i + 1 + n)
        plt.imshow(noisy[i].squeeze(), cmap='gray')
        plt.axis('off')
        if i == 0: plt.title("Noisy")

        # Reconstructed from original images
        plt.subplot(4, n, i + 1 + 2 * n)
        plt.imshow(reconstructed[i].squeeze(), cmap='gray')
        plt.axis('off')
        if i == 0: plt.title("Reconstructed (Clean)")

        # Reconstructed from noisy images
        plt.subplot(4, n, i + 1 + 3 * n)
        plt.imshow(reconstructed[i].squeeze(), cmap='gray')
        plt.axis('off')
        if i == 0: plt.title("Reconstructed (Noisy)")
    plt.show()


reconstructed_test = autoencoder.predict(X_test_noisy)

plot_images(X_test, X_test_noisy, reconstructed_test, n=5)
