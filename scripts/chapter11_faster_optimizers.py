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


# backend
K = keras.backend


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


# power scheduling
def power_decay(_lr0, _decay, _n_steps_per_epoch):
    def _power_decay_fn(epoch):
        return _lr0 / (1 + epoch * _n_steps_per_epoch * _decay)
    return _power_decay_fn


# exponential scheduling
def exponential_decay(lr0, s):  # s: the number of batches in one epoch, maybe 20
    def exponential_decay_fn(epoch):
        return lr0 * 0.1**(epoch / s)
    return exponential_decay_fn


# update lr at each iteration (batch) NOT epoch, so need to re-write the callback class
class ExponentialDecay(keras.callbacks.Callback):
    def __init__(self, s=40000):
        super().__init__()
        self.s = s  # s here is not the number of batches in one epoch but the total number of batches in training

    def on_batch_begin(self, batch, logs=None):
        # Note: the `batch` argument is reset at each epoch
        lr = K.get_value(self.model.optimizer.lr)
        K.set_value(self.model.optimizer.lr, lr * 0.1**(1 / self.s))  # change s into self.s

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)


def piecewise_constant_fn(epoch):
    if epoch < 5:
        return 0.01
    elif epoch < 15:
        return 0.005
    else:
        return 0.001


def piecewise_constant(boundaries, values):
    boundaries = np.array([0] + boundaries)
    values = np.array(values)

    def piecewise_constant_fn(epoch):
        return values[np.argmax(boundaries > epoch) - 1]
    return piecewise_constant_fn


class ExponentialLearningRate(keras.callbacks.Callback):
    def __init__(self, factor):
        self.factor = factor
        self.rates = []
        self.losses = []

    def on_batch_end(self, batch, logs):
        self.rates.append(K.get_value(self.model.optimizer.lr))
        self.losses.append(logs["loss"])
        K.set_value(self.model.optimizer.lr, self.model.optimizer.lr * self.factor)


def find_learning_rate(model, X, y, epochs=1, batch_size=32, min_rate=10**-5, max_rate=10):
    init_weights = model.get_weights()
    iterations = len(X) // batch_size * epochs
    factor = np.exp(np.log(max_rate / min_rate) / iterations)
    init_lr = K.get_value(model.optimizer.lr)
    K.set_value(model.optimizer.lr, min_rate)
    exp_lr = ExponentialLearningRate(factor)
    history = model.fit(X, y, epochs=epochs, batch_size=batch_size,
                        callbacks=[exp_lr])
    K.set_value(model.optimizer.lr, init_lr)
    model.set_weights(init_weights)
    return exp_lr.rates, exp_lr.losses


def plot_lr_vs_loss(rates, losses):
    plt.plot(rates, losses)
    plt.gca().set_xscale('log')
    plt.hlines(min(losses), min(rates), max(rates))
    plt.axis([min(rates), max(rates), min(losses), (losses[0] + min(losses)) / 2])
    plt.xlabel("Learning rate")
    plt.ylabel("Loss")


class OneCycleScheduler(keras.callbacks.Callback):
    def __init__(self, iterations, max_rate, start_rate=None,
                 last_iterations=None, last_rate=None):
        self.iterations = iterations
        self.max_rate = max_rate
        self.start_rate = start_rate or max_rate / 10
        self.last_iterations = last_iterations or iterations // 10 + 1
        self.half_iteration = (iterations - self.last_iterations) // 2
        self.last_rate = last_rate or self.start_rate / 1000
        self.iteration = 0

    def _interpolate(self, iter1, iter2, rate1, rate2):
        return ((rate2 - rate1) * (self.iteration - iter1)
                / (iter2 - iter1) + rate1)

    def on_batch_begin(self, batch, logs):
        if self.iteration < self.half_iteration:
            rate = self._interpolate(0, self.half_iteration, self.start_rate, self.max_rate)
        elif self.iteration < 2 * self.half_iteration:
            rate = self._interpolate(self.half_iteration, 2 * self.half_iteration,
                                     self.max_rate, self.start_rate)
        else:
            rate = self._interpolate(2 * self.half_iteration, self.iterations,
                                     self.start_rate, self.last_rate)
            rate = max(rate, self.last_rate)
        self.iteration += 1
        K.set_value(self.model.optimizer.lr, rate)


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
    optimizer = keras.optimizers.SGD(lr=0.01, decay=1e-4)  # set the decay value to make the power schedule

    # build model
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[28, 28]),
        keras.layers.Dense(300, activation="selu", kernel_initializer="lecun_normal"),
        keras.layers.Dense(100, activation="selu", kernel_initializer="lecun_normal"),
        keras.layers.Dense(10, activation="softmax")
    ])
    model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    n_epochs = 25
    power_schedule_history = model.fit(X_train_scaled, y_train, epochs=n_epochs,
                                       validation_data=(X_valid_scaled, y_valid))
    # loss: 0.2034 - accuracy: 0.9281 - val_loss: 0.3142 - val_accuracy: 0.8900

    model.evaluate(X_test, y_test)  # loss: 0.6491 - accuracy: 0.7272

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
    plt.title("Keras's Power Scheduling", fontsize=14)
    plt.grid(True)

    save_fig("power_scheduling")
    # plt.show()

    # my own power scheduler
    lr0 = 0.01
    decay = 1e-4
    batch_size = 32
    n_steps_per_epoch = len(X_train) // batch_size

    power_decay_fn = power_decay(_lr0=lr0, _decay=decay, _n_steps_per_epoch=n_steps_per_epoch)
    my_optimizer = keras.optimizers.SGD(lr=0.01)  # without decay defined here, so it's not a power schedule in Keras

    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[28, 28]),
        keras.layers.Dense(300, activation="selu", kernel_initializer="lecun_normal"),
        keras.layers.Dense(100, activation="selu", kernel_initializer="lecun_normal"),
        keras.layers.Dense(10, activation="softmax")
    ])
    model.compile(loss="sparse_categorical_crossentropy", optimizer=my_optimizer, metrics=["accuracy"])

    lr_scheduler = keras.callbacks.LearningRateScheduler(power_decay_fn)

    n_epochs = 25
    my_power_scheduler_history = model.fit(X_train_scaled, y_train, epochs=n_epochs,
                                           validation_data=(X_valid_scaled, y_valid),
                                           callbacks=[lr_scheduler])  # add lr scheduler to make power scheduler
    # loss: 0.2022 - accuracy: 0.9295 - val_loss: 0.3193 - val_accuracy: 0.8874

    model.evaluate(X_test, y_test)  # loss: 0.6994 - accuracy: 0.6830 It's not good ...

    # plot
    plt.figure(figsize=(6.4, 4.8))
    plt.plot(my_power_scheduler_history.epoch, my_power_scheduler_history.history["lr"], "o-")
    plt.axis([0, n_epochs - 1, 0, 0.011])
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("My Power Scheduling", fontsize=14)
    plt.grid(True)
    # plt.show()

    # Exponential Scheduling

    # callback: exponential schedule
    exponential_decay_fn = exponential_decay(lr0=0.01, s=20)  # if s=n_epochs = 25 that's ok

    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[28, 28]),
        keras.layers.Dense(300, activation="selu", kernel_initializer="lecun_normal"),
        keras.layers.Dense(100, activation="selu", kernel_initializer="lecun_normal"),
        keras.layers.Dense(10, activation="softmax")
    ])
    model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])  # not self-defined
    n_epochs = 25

    lr_scheduler = keras.callbacks.LearningRateScheduler(exponential_decay_fn)
    exponential_schedule_history = model.fit(X_train_scaled, y_train, epochs=n_epochs,
                                             validation_data=(X_valid_scaled, y_valid),
                                             callbacks=[lr_scheduler])  # add exponential schedule call back

    # plot
    plt.figure(figsize=(6.4, 4.8))
    plt.plot(exponential_schedule_history.epoch, exponential_schedule_history.history["lr"], "o-")
    plt.axis([0, n_epochs - 1, 0, 0.011])
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Exponential Scheduling", fontsize=14)
    plt.grid(True)
    # plt.show()

    # rewrite the callback class to update lr at each batch not epoch
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[28, 28]),
        keras.layers.Dense(300, activation="selu", kernel_initializer="lecun_normal"),
        keras.layers.Dense(100, activation="selu", kernel_initializer="lecun_normal"),
        keras.layers.Dense(10, activation="softmax")
    ])
    lr0 = 0.01
    optimizer = keras.optimizers.Nadam(lr=lr0)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    n_epochs = 25

    s = 20 * len(X_train) // 32  # number of steps in 20 epochs (batch size = 32) also s==1/decay
    exp_decay = ExponentialDecay(s)  # rewrite the callback class to update lr at each batch not epoch
    history = model.fit(X_train_scaled, y_train, epochs=n_epochs,
                        validation_data=(X_valid_scaled, y_valid),
                        callbacks=[exp_decay])

    # plot
    n_steps = n_epochs * len(X_train) // 32  # the total steps of all epochs
    steps = np.arange(n_steps)
    lrs = lr0 * 0.1 ** (steps / s)

    plt.figure(figsize=(6.4, 4.8))
    plt.plot(steps, lrs, "-", linewidth=2)
    plt.axis([0, n_steps - 1, 0, lr0 * 1.1])
    plt.xlabel("Batch")
    plt.ylabel("Learning Rate")
    plt.title("Exponential Scheduling (per batch)", fontsize=14)
    plt.grid(True)
    plt.show()

    # Piecewise Constant Scheduling

    # callback: define piece-wise constant scheduling callback
    piecewise_constant_fn = piecewise_constant([5, 15], [0.01, 0.005, 0.001])
    lr_scheduler = keras.callbacks.LearningRateScheduler(piecewise_constant_fn)

    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[28, 28]),
        keras.layers.Dense(300, activation="selu", kernel_initializer="lecun_normal"),
        keras.layers.Dense(100, activation="selu", kernel_initializer="lecun_normal"),
        keras.layers.Dense(10, activation="softmax")
    ])
    model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])

    n_epochs = 25
    piece_wise_constant_history = model.fit(X_train_scaled, y_train, epochs=n_epochs,
                                            validation_data=(X_valid_scaled, y_valid),
                                            callbacks=[lr_scheduler])

    # plot
    plt.figure(figsize=(6.4, 4.8))
    plt.plot(history.epoch, [piecewise_constant_fn(epoch) for epoch in history.epoch], "o-")
    plt.axis([0, n_epochs - 1, 0, 0.011])
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Piecewise Constant Scheduling", fontsize=14)
    plt.grid(True)
    # plt.show()

    # Performance Scheduling

    tf.random.set_random_seed(42)
    np.random.seed(42)

    lr_scheduler = keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)  # reduce learning rate on plateau

    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[28, 28]),
        keras.layers.Dense(300, activation="selu", kernel_initializer="lecun_normal"),
        keras.layers.Dense(100, activation="selu", kernel_initializer="lecun_normal"),
        keras.layers.Dense(10, activation="softmax")
    ])
    optimizer = keras.optimizers.SGD(lr=0.02, momentum=0.9)  # momentum
    model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    n_epochs = 25
    performance_schedule_history = model.fit(X_train_scaled, y_train, epochs=n_epochs,
                                             validation_data=(X_valid_scaled, y_valid),
                                             callbacks=[lr_scheduler])

    # plot
    plt.figure(figsize=(6.4, 4.8))
    lr_scheduler = keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)

    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[28, 28]),
        keras.layers.Dense(300, activation="selu", kernel_initializer="lecun_normal"),
        keras.layers.Dense(100, activation="selu", kernel_initializer="lecun_normal"),
        keras.layers.Dense(10, activation="softmax")
    ])
    optimizer = keras.optimizers.SGD(lr=0.02, momentum=0.9)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    n_epochs = 25
    history = model.fit(X_train_scaled, y_train, epochs=n_epochs,
                        validation_data=(X_valid_scaled, y_valid),
                        callbacks=[lr_scheduler])

    # tf.keras API

    # Exponential Scheduling
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[28, 28]),
        keras.layers.Dense(300, activation="selu", kernel_initializer="lecun_normal"),
        keras.layers.Dense(100, activation="selu", kernel_initializer="lecun_normal"),
        keras.layers.Dense(10, activation="softmax")
    ])

    s = 20 * len(X_train) // 32  # number of steps in 20 epochs (batch size = 32), a big number for all epochs.
    learning_rate = keras.optimizers.schedules.ExponentialDecay(0.01, s, 0.1)
    optimizer = keras.optimizers.SGD(learning_rate)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    n_epochs = 25
    tf_keras_api_history = model.fit(X_train_scaled, y_train, epochs=n_epochs,
                                     validation_data=(X_valid_scaled, y_valid))

    # Piecewise Constant Scheduling
    learning_rate = keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=[5. * n_steps_per_epoch, 15. * n_steps_per_epoch],
        values=[0.01, 0.005, 0.001])

    # 1Cycle scheduling

    tf.random.set_random_seed(42)
    np.random.seed(42)

    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[28, 28]),
        keras.layers.Dense(300, activation="selu", kernel_initializer="lecun_normal"),
        keras.layers.Dense(100, activation="selu", kernel_initializer="lecun_normal"),
        keras.layers.Dense(10, activation="softmax")
    ])
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=keras.optimizers.SGD(lr=1e-3),
                  metrics=["accuracy"])

    batch_size = 128
    rates, losses = find_learning_rate(model, X_train_scaled, y_train, epochs=1, batch_size=batch_size)
    plot_lr_vs_loss(rates, losses)

    n_epochs = 25
    onecycle = OneCycleScheduler(len(X_train) // batch_size * n_epochs, max_rate=0.05)
    onecycle_history = model.fit(X_train_scaled, y_train, epochs=n_epochs, batch_size=batch_size,
                                 validation_data=(X_valid_scaled, y_valid),
                                 callbacks=[onecycle])