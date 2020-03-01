# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

import tensorflow as tf
from tensorflow import keras
print(tf.__version__)  # 2.0.1
print(keras.__version__)  # 2.2.4-tf

# Common imports
import numpy as np
import os
from matplotlib.colors import ListedColormap
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# chapter imports
from keras import layers
from keras import models
from keras.datasets import mnist
from keras.utils import to_categorical

# to make this notebook's output stable across runs
np.random.seed(42)

# To plot pretty figures
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Ignore useless warnings (see SciPy issue #5998)
import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")

# Where to save the figures
PROJECT_ROOT_DIR = "D:\\AI\\handson-ml2-master\\"
CHAPTER_ID = "ann"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)
LOGS_PATH = os.path.join(PROJECT_ROOT_DIR, "tensorboard_logs", CHAPTER_ID)
os.makedirs(LOGS_PATH, exist_ok=True)


def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(LOGS_PATH, run_id)


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", path)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


class WideAndDeepModel(keras.models.Model):
    def __init__(self, units=30, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.hidden1 = keras.layers.Dense(units, activation=activation)
        self.hidden2 = keras.layers.Dense(units, activation=activation)
        self.main_output = keras.layers.Dense(1)
        self.aux_output = keras.layers.Dense(1)

    def call(self, inputs):  # call() defines the dynamic model's logic
        input_A, input_B = inputs
        hidden1 = self.hidden1(input_B)
        hidden2 = self.hidden2(hidden1)
        concat = keras.layers.concatenate([input_A, hidden2])
        main_output = self.main_output(concat)
        aux_output = self.aux_output(hidden2)
        return main_output, aux_output


class PrintValTrainRatioCallback(keras.callbacks.Callback):  # define my own callbacks
    def on_epoch_end(self, epoch, logs):
        print("\nval/train: {:.2f}".format(logs["val_loss"] / logs["loss"]))  # to print losses at the end of each epoch


class ExponentialLearningRate(keras.callbacks.Callback):
    def __init__(self, factor):
        self.factor = factor
        self.rates = []
        self.losses = []

    def on_batch_end(self, batch, logs):
        self.rates.append(K.get_value(self.model.optimizer.lr))
        self.losses.append(logs["loss"])
        K.set_value(self.model.optimizer.lr, self.model.optimizer.lr * self.factor)


if __name__ == '__main__':

    # 10.

    # data set
    (X_train_full, y_train_full), (X_test, y_test) = keras.datasets.mnist.load_data()
    print(X_train_full.shape, X_train_full.dtype)

    X_valid, X_train = X_train_full[:5000] / 255., X_train_full[5000:] / 255.
    y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
    X_test = X_test / 255.

    # plot a sample
    plt.imshow(X_train[0], cmap="binary")
    plt.axis('off')
    # plt.show()

    # plot samples
    n_rows = 4
    n_cols = 10
    plt.figure(figsize=(n_cols * 1.2, n_rows * 1.2))
    for row in range(n_rows):
        for col in range(n_cols):
            index = n_cols * row + col
            plt.subplot(n_rows, n_cols, index + 1)
            plt.imshow(X_train[index], cmap="binary", interpolation="nearest")
            plt.axis('off')
            plt.title(y_train[index], fontsize=12)
    plt.subplots_adjust(wspace=0.2, hspace=0.5)
    # plt.show()

    K = keras.backend
    keras.backend.clear_session()
    np.random.seed(42)
    tf.random.set_random_seed(42)

    # build model
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[28, 28]),
        keras.layers.Dense(300, activation="relu"),
        keras.layers.Dense(100, activation="relu"),
        keras.layers.Dense(10, activation="softmax")
    ])

    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=keras.optimizers.SGD(lr=2e-1),
                  metrics=["accuracy"])
    expon_lr = ExponentialLearningRate(factor=1.005)

    history = model.fit(X_train, y_train, epochs=1,
                        validation_data=(X_valid, y_valid),
                        callbacks=[expon_lr])

    # plot the loss as a function of the learning rate
    plt.figure(figsize=(10, 8))
    plt.plot(expon_lr.rates, expon_lr.losses)
    plt.gca().set_xscale('log')
    plt.hlines(min(expon_lr.losses), min(expon_lr.rates), max(expon_lr.rates))
    plt.axis([min(expon_lr.rates), max(expon_lr.rates), 0, expon_lr.losses[0]])
    plt.xlabel("Learning rate")
    plt.ylabel("Loss")
    plt.plot(expon_lr.rates, expon_lr.losses)
    plt.gca().set_xscale('log')
    plt.hlines(min(expon_lr.losses), min(expon_lr.rates), max(expon_lr.rates))
    plt.axis([min(expon_lr.rates), max(expon_lr.rates), 0, expon_lr.losses[0]])
    plt.xlabel("Learning rate")
    plt.ylabel("Loss")
    plt.show()

    # build a new model
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[28, 28]),
        keras.layers.Dense(300, activation="relu"),
        keras.layers.Dense(100, activation="relu"),
        keras.layers.Dense(10, activation="softmax")
    ])
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=keras.optimizers.SGD(lr=2e-1),
                  metrics=["accuracy"])

    # tensorboard show
    run_logdir = get_run_logdir()

    # callback
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=20)
    checkpoint_cb = keras.callbacks.ModelCheckpoint("D:/AI/handson-ml2-master/h5/chapter10_exercise.h5", save_best_only=True)
    tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)

    history = model.fit(X_train, y_train, epochs=100, validation_data=(X_valid, y_valid),
                        callbacks=[early_stopping_cb, checkpoint_cb, tensorboard_cb])

    model = keras.models.load_model("D:/AI/handson-ml2-master/h5/chapter10_exercise.h5")  # rollback to best model
    model.evaluate(X_test, y_test)  # loss: 0.0350 - accuracy: 0.9812 not 99% yet.

