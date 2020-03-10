# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

# TensorFlow ≥2.0 is required
import tensorflow as tf
from tensorflow import keras
assert tf.__version__ >= "2.0"

# Common imports
import numpy as np
import os

# Chapter imports

# to make this notebook's output stable across runs
np.random.seed(42)

# To plot pretty figures
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the files
PROJECT_ROOT_DIR = "D:\\AI\\handson-ml2-master\\"
CHAPTER_ID = "deep"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)
LOGS_PATH = os.path.join(PROJECT_ROOT_DIR, "tensorboard_logs", CHAPTER_ID)
os.makedirs(LOGS_PATH, exist_ok=True)
H5_PATH = os.path.join(PROJECT_ROOT_DIR, 'h5', CHAPTER_ID)
os.makedirs(H5_PATH, exist_ok=True)


def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(LOGS_PATH, run_id)


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


def save_model(model, model_id, h5_extension="h5"):
    path = os.path.join(H5_PATH, model_id + "." + h5_extension)
    print("Saving model", model_id)
    model.save(path)


def split_dataset(X, y):
    y_5_or_6 = (y == 5) | (y == 6) # sandals or shirts
    y_A = y[~y_5_or_6]  # NOT 5 or 6
    y_A[y_A > 6] -= 2 # class indices 7, 8, 9 should be moved to 5, 6, 7
    y_B = (y[y_5_or_6] == 6).astype(np.float32) # If shirt (==6), y_B[i]=1, or =0
    return ((X[~y_5_or_6], y_A),
            (X[y_5_or_6], y_B))


if __name__ == '__main__':

    # Momentum Optimization

    optimizer_momentum = keras.optimizers.SGD(lr=0.001, momentum=0.9)  # It's easy to add momentum optimizer. Just add this para.

    # Nesterov Accelerated Gradient - NAG

    optimizer_nesterov = keras.optimizers.SGD(lr=0.001, momentum=0.9, nesterov=True)

    # AdaGrad

    optimizer_adagrad = keras.optimizers.Adagrad(lr=0.001)  # lr: learning rate

    # RMSProp

    optimizer_rmsprop = keras.optimizers.RMSprop(lr=0.001, rho=0.9)  # rho: the β mentioned in book
    # prefer this RMSProp than AdaGrad above. Refer to cloud note.

    # Adam Optimization

    optimizer_adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)

    # Adamax Optimization

    optimizer_adamax = keras.optimizers.Adamax(lr=0.001, beta_1=0.9, beta_2=0.999)

    # Nadam Optimization

    optimizer_nadam = keras.optimizers.Nadam(lr=0.001, beta_1=0.9, beta_2=0.999)

    # Learning Rate Scheduling

    # Power Scheduling

    # formula: lr = lr0 / (1 + steps / s)**c  Keras uses c=1 and s = 1 / decay
    # so lr = lr0 / (1 + steps*decay)

    # data set
    (X_train_full, y_train_full), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
    X_train_full = X_train_full / 255.0
    X_test = X_test / 255.0
    X_valid, X_train = X_train_full[:5000], X_train_full[5000:]
    y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

    pixel_means = X_train.mean(axis=0, keepdims=True)
    pixel_stds = X_train.std(axis=0, keepdims=True)
    X_train_scaled = (X_train - pixel_means) / pixel_stds
    X_valid_scaled = (X_valid - pixel_means) / pixel_stds
    X_test_scaled = (X_test - pixel_means) / pixel_stds

    # optimizer
    optimizer = keras.optimizers.SGD(lr=0.01, decay=1e-4)

    # build model
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[28, 28]),
        keras.layers.Dense(300, activation="selu", kernel_initializer="lecun_normal"),
        keras.layers.Dense(100, activation="selu", kernel_initializer="lecun_normal"),
        keras.layers.Dense(10, activation="softmax")
    ])
    model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    n_epochs = 25
    history = model.fit(X_train_scaled, y_train, epochs=n_epochs,
                        validation_data=(X_valid_scaled, y_valid))

    # plot
    learning_rate = 0.01
    decay = 1e-4
    batch_size = 32
    n_steps_per_epoch = len(X_train) // batch_size
    epochs = np.arange(n_epochs)
    lrs = learning_rate / (1 + decay * epochs * n_steps_per_epoch)  # the formula above: lr = lr0 / (1 + steps*decay)

    plt.plot(epochs, lrs, "o-")
    plt.axis([0, n_epochs - 1, 0, 0.01])
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Power Scheduling", fontsize=14)
    plt.grid(True)

    save_fig("power_scheduling")
    # plt.show()

    # Exponential Scheduling