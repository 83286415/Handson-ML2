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

    # Reusing a Keras model

    print('Reusing a Keras model')

    # data set: Fashion MNIST
    (X_train_full, y_train_full), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
    X_train_full = X_train_full / 255.0
    X_test = X_test / 255.0
    X_valid, X_train = X_train_full[:5000], X_train_full[5000:]
    y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

    (X_train_A, y_train_A), (X_train_B, y_train_B) = split_dataset(X_train, y_train)  # A: not 5or6;     B: 5or6
    (X_valid_A, y_valid_A), (X_valid_B, y_valid_B) = split_dataset(X_valid, y_valid)
    (X_test_A, y_test_A), (X_test_B, y_test_B) = split_dataset(X_test, y_test)
    X_train_B = X_train_B[:200]
    y_train_B = y_train_B[:200]

    print('A shape: ', X_train_A.shape)  # (43986, 28, 28)
    print('B shape: ', X_train_B.shape)  # (200, 28, 28)
    print('A 30 labels: ', y_train_A[:30])  # [4 0 5 7 7 7 4 4 3 4 0 1 6 3 4 3 2 6 5 3 4 5 1 3 4 2 0 6 7 1]
    print('B 30 labels: ', y_train_B[:30])
    # [1. 1. 0. 0. 0. 0. 1. 1. 1. 0. 0. 1. 1. 0. 0. 0. 0. 0. 0. 1. 1. 0. 0. 1. 1. 0. 1. 1. 1. 1.]

    # build model A
    model_A = keras.models.Sequential()
    model_A.add(keras.layers.Flatten(input_shape=[28, 28]))
    for n_hidden in (300, 100, 50, 50, 50):  # add 5 layers
        model_A.add(keras.layers.Dense(n_hidden, activation="selu"))
    model_A.add(keras.layers.Dense(8, activation="softmax"))  # multi-classification

    model_A.compile(loss="sparse_categorical_crossentropy",  # multi-classification
                    optimizer=keras.optimizers.SGD(lr=1e-3),
                    metrics=["accuracy"])

    a_history = model_A.fit(X_train_A, y_train_A, epochs=20, validation_data=(X_valid_A, y_valid_A))
    # loss: 0.2184 - accuracy: 0.9256 - val_loss: 0.2346 - val_accuracy: 0.9180

    # save model A
    save_model(model_A, 'my_model_A')

    # build model B
    model_B = keras.models.Sequential()
    model_B.add(keras.layers.Flatten(input_shape=[28, 28]))
    for n_hidden in (300, 100, 50, 50, 50):
        model_B.add(keras.layers.Dense(n_hidden, activation="selu"))
    model_B.add(keras.layers.Dense(1, activation="sigmoid"))  # binary classification

    model_B.compile(loss="binary_crossentropy",  # binary classification
                    optimizer=keras.optimizers.SGD(lr=1e-3),
                    metrics=["accuracy"])

    b_history = model_B.fit(X_train_B, y_train_B, epochs=20, validation_data=(X_valid_B, y_valid_B))
    # loss: 0.0806 - accuracy: 0.9950 - val_loss: 0.1098 - val_accuracy: 0.9817

    print(model_B.summary())
    '''_________________________________________________________________
        Layer (type)                 Output Shape              Param #   
        =================================================================
        flatten_1 (Flatten)          (None, 784)               0         
        _________________________________________________________________
        dense_6 (Dense)              (None, 300)               235500    
        _________________________________________________________________
        dense_7 (Dense)              (None, 100)               30100     
        _________________________________________________________________
        dense_8 (Dense)              (None, 50)                5050      
        _________________________________________________________________
        dense_9 (Dense)              (None, 50)                2550      
        _________________________________________________________________
        dense_10 (Dense)             (None, 50)                2550      
        _________________________________________________________________
        dense_11 (Dense)             (None, 1)                 51        
        =================================================================
        Total params: 275,801
        Trainable params: 275,801
        Non-trainable params: 0
        _________________________________________________________________'''

    # share layers between A and B_A: if we fit B_A, the A will be fitted also.
    model_A = keras.models.load_model("D:/AI/handson-ml2-master/h5/deep/my_model_A.h5")
    model_B_on_A = keras.models.Sequential(model_A.layers[:-1])  # remove the softmax, topest layer from A
    model_B_on_A.add(keras.layers.Dense(1, activation="sigmoid"))  # add the binary classification output layer

    # to avoid affect model A, so clone it and set its weights
    model_A_clone = keras.models.clone_model(model_A)
    model_A_clone.set_weights(model_A.get_weights())  # make A clone

    for layer in model_B_on_A.layers[:-1]:  # freeze(fixed) the BA model except its topest layer, sigmoid layer.
        layer.trainable = False

    # ALWAYS compile model after freezing or unfreezing it!!!
    model_B_on_A.compile(loss="binary_crossentropy",
                         optimizer=keras.optimizers.SGD(lr=1e-3),
                         metrics=["accuracy"])

    # freeze reused layers and fit the model to make new layer to learn the reasonable weights.
    history_B_on_A = model_B_on_A.fit(X_train_B, y_train_B, epochs=4, validation_data=(X_valid_B, y_valid_B))
    # loss: 0.3743 - accuracy: 0.8500 - val_loss: 0.3978 - val_accuracy: 0.8367

    # unfreeze layers and fine tune the reused layers for the new task B
    for layer in model_B_on_A.layers[:-1]:  # unfreeze the BA model's layers
        layer.trainable = True

    # ALWAYS compile model after freezing or unfreezing it!!!
    model_B_on_A.compile(loss="binary_crossentropy",
                         optimizer=keras.optimizers.SGD(lr=1e-4),  # reduce the learning rate to protect reused weights.
                         metrics=["accuracy"])

    # to fit again to fine tune the weights according to the task B
    history_B_on_A = model_B_on_A.fit(X_train_B, y_train_B, epochs=16, validation_data=(X_valid_B, y_valid_B))
    # loss: 0.0551 - accuracy: 1.0000 - val_loss: 0.0745 - val_accuracy: 0.9929  much better than all models before

    # evaluate
    model_B.evaluate(X_test_B, y_test_B)  # loss: 0.0964 - accuracy: 0.9900
    model_B_on_A.evaluate(X_test_B, y_test_B)  # loss: 0.0756 - accuracy: 0.9910  better than model_B

    # Note: the evaluation on model_B_on_A is better. But this result is fake!!! Because transfer learning is NOT good
    # for small dense network, do NOT use it for dense network anymore! It works best with deep convolutional neural net
