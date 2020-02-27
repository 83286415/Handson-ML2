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
import pandas as pd

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


if __name__ == '__main__':

    # data set
    housing = fetch_california_housing()

    X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, random_state=42)  # default 0.25

    # scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)
    X_test = scaler.transform(X_test)

    X_train_A, X_train_B = X_train[:, :5], X_train[:, 2:]  # split train data into two input
    X_valid_A, X_valid_B = X_valid[:, :5], X_valid[:, 2:]
    X_test_A, X_test_B = X_test[:, :5], X_test[:, 2:]
    X_new_A, X_new_B = X_test_A[:3], X_test_B[:3]  # split test data into two input
    X_new = X_test[:3]

    # Saving and Restoring a Model

    # save and save_wight: https://blog.csdn.net/leviopku/article/details/86612293
    print('Saving and Restoring a Model...')
    np.random.seed(42)
    tf.random.set_random_seed(42)

    # build a model
    model = keras.models.Sequential([
        keras.layers.Dense(30, activation="relu", input_shape=[8]),
        keras.layers.Dense(30, activation="relu"),
        keras.layers.Dense(1)
    ])

    # save and load
    model.save('D:/AI/handson-ml2-master/h5/sequential_without_fitting.h5')  # save before fitting
    # model.save_weight('sequential_without_fitting.ckpt')  # save weight before fitting, a checkpoint file

    model.compile(loss="mse", optimizer=keras.optimizers.SGD(lr=1e-3))
    model.fit(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid))
    mse_test = model.evaluate(X_test, y_test)
    model.save("D:/AI/handson-ml2-master/h5/sequential_after_fitting.h5")  # save after fitting
    # model.save_weight('sequential_after_fitting.ckpt')  # save weight after fitting, a checkpoint file

    model = keras.models.load_model("D:/AI/handson-ml2-master/h5/sequential_after_fitting.h5")
    y_pred = model.predict(X_new)
    print(y_pred)  # [[0.54909724] [1.6584849 ] [3.0271604 ]]
    print(model.summary())
    '''Model: "sequential"
        _________________________________________________________________
        Layer (type)                 Output Shape              Param #   
        =================================================================
        dense (Dense)                (None, 30)                270       
        _________________________________________________________________
        dense_1 (Dense)              (None, 30)                930       
        _________________________________________________________________
        dense_2 (Dense)              (None, 1)                 31        
        =================================================================
        Total params: 1,231
        Trainable params: 1,231
        Non-trainable params: 0
        _________________________________________________________________'''

    model = keras.models.load_model('D:/AI/handson-ml2-master/h5/sequential_without_fitting.h5')
    print(model.summary())
    '''Model: "sequential"
        _________________________________________________________________
        Layer (type)                 Output Shape              Param #   
        =================================================================
        dense (Dense)                (None, 30)                270       
        _________________________________________________________________
        dense_1 (Dense)              (None, 30)                930       
        _________________________________________________________________
        dense_2 (Dense)              (None, 1)                 31        
        =================================================================
        Total params: 1,231
        Trainable params: 1,231
        Non-trainable params: 0
        _________________________________________________________________'''
    # cannot use model.summary to tell the difference between this h5 and that below h5.
    # need the tensorboard to show the difference.

    # model = keras.models.load_model('sequential_after_fitting.ckpt')  # Sequential API doesn't support save_weight()
    # print(model.summary())

    # Using Callbacks

    print('Using Callbacks...')
    keras.backend.clear_session()
    np.random.seed(42)
    tf.random.set_random_seed(42)

    # build a model
    model = keras.models.Sequential([
        keras.layers.Dense(30, activation="relu", input_shape=[8]),
        keras.layers.Dense(30, activation="relu"),
        keras.layers.Dense(1)
    ])

    # saves the best model in each training interval
    model.compile(loss="mse", optimizer=keras.optimizers.SGD(lr=1e-3))
    checkpoint_cb = keras.callbacks.ModelCheckpoint("D:/AI/handson-ml2-master/h5/my_keras_model.h5", save_best_only=True)

    history = model.fit(X_train, y_train, epochs=10,
                        validation_data=(X_valid, y_valid),
                        callbacks=[checkpoint_cb])  # add callbacks list to save the best model
    model = keras.models.load_model("D:/AI/handson-ml2-master/h5/my_keras_model.h5")  # rollback to best model
    mse_test = model.evaluate(X_test, y_test)
    print(mse_test)  # 0.43821001052856445

    # early stopping used to reduce the training process time
    model.compile(loss="mse", optimizer=keras.optimizers.SGD(lr=1e-3))
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)  # roll back to best model
    #  patience: number of epochs with no improvement after which training will be stopped.
    history = model.fit(X_train, y_train, epochs=100,
                        validation_data=(X_valid, y_valid),
                        callbacks=[checkpoint_cb, early_stopping_cb])  # callbacks list: combine two callbacks
    mse_test = model.evaluate(X_test, y_test)  # stopped at the end of epoch 57!!! save training time!
    print(mse_test)  # 0.34666288180868754

    # self define a callback
    val_train_ratio_cb = PrintValTrainRatioCallback()  # define a simple callback to print losses on epoch end
    history = model.fit(X_train, y_train, epochs=3,
                        validation_data=(X_valid, y_valid),
                        callbacks=[val_train_ratio_cb])
    # output 'val/train: 0.94' at the end of each epoch.

    # TensorBoard

    # usage:
    # 1. run this code
    # 2. open a shell under dir:  D:\AI\handson - ml2 - master\tensorboard_logs\ann
    # 3. input the cmd on shell:tensorboard - -port = 6606 - -logdir = D:\AI\handson - ml2 - master\tensorboard_logs\ann
    # 4. open a web with this address: localhost:6606

    # log dir
    run_logdir = get_run_logdir()
    print(run_logdir)  # D:\AI\handson-ml2-master\tensorboard_logs\ann\run_2020_02_27-16_31_57

    keras.backend.clear_session()
    np.random.seed(42)
    tf.random.set_random_seed(42)

    # build a model with learning rate = 0.001
    model = keras.models.Sequential([
        keras.layers.Dense(30, activation="relu", input_shape=[8]),
        keras.layers.Dense(30, activation="relu"),
        keras.layers.Dense(1)
    ])
    model.compile(loss="mse", optimizer=keras.optimizers.SGD(lr=1e-3))

    # tensorboard
    tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)  # a tensorboard callback
    history = model.fit(X_train, y_train, epochs=30,
                        validation_data=(X_valid, y_valid),
                        callbacks=[checkpoint_cb, tensorboard_cb])

    # make a new log dir
    run_logdir2 = get_run_logdir()
    print(run_logdir2)

    keras.backend.clear_session()
    np.random.seed(42)
    tf.random.set_random_seed(42)

    # model with a learning rate = 0.05, which is bigger than before 0.001
    model = keras.models.Sequential([
        keras.layers.Dense(30, activation="relu", input_shape=[8]),
        keras.layers.Dense(30, activation="relu"),
        keras.layers.Dense(1)
    ])
    model.compile(loss="mse", optimizer=keras.optimizers.SGD(lr=0.05))

    # tensorboard
    tensorboard_cb = keras.callbacks.TensorBoard(run_logdir2)
    history = model.fit(X_train, y_train, epochs=30,
                        validation_data=(X_valid, y_valid),
                        callbacks=[checkpoint_cb, tensorboard_cb])  # four more curves shown on tensorboard

    # help
    help(keras.callbacks.TensorBoard.__init__)
    '''Help on function __init__ in module tensorflow.python.keras.callbacks:

        __init__(self, log_dir='logs', histogram_freq=0, write_graph=True, write_images=False, update_freq='epoch', 
        profile_batch=2, embeddings_freq=0, embeddings_metadata=None, **kwargs)
            Initialize self.  See help(type(self)) for accurate signature.'''
