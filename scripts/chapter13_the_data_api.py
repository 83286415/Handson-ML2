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
CHAPTER_ID = "data"
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
    if model_id.endswith('h5'):
        path = os.path.join(H5_PATH, model_id)
    else:
        path = os.path.join(H5_PATH, model_id + "." + h5_extension)
    print("Saving model", path)
    model.save(path)


def load_model(model_id, **kwargs):
    if not model_id.endswith('h5'):
        model_id = model_id + ".h5"
    if not os.path.exists(model_id):
        model_id = os.path.join(H5_PATH, model_id)
    return keras.models.load_model(model_id, **kwargs)


if __name__ == '__main__':

    # Datasets

    X = tf.range(10)
    dataset = tf.data.Dataset.from_tensor_slices(X)  # return a slice of tensors, that is a list maybe
    # equal to: dataset_new = tf.data.Dataset.range(10)

    for item in dataset:
        print(item)
        ''' tf.Tensor(0, shape=(), dtype=int32)
            tf.Tensor(1, shape=(), dtype=int32)
            tf.Tensor(2, shape=(), dtype=int32)
            tf.Tensor(3, shape=(), dtype=int32)
            tf.Tensor(4, shape=(), dtype=int32)
            tf.Tensor(5, shape=(), dtype=int32)
            tf.Tensor(6, shape=(), dtype=int32)
            tf.Tensor(7, shape=(), dtype=int32)
            tf.Tensor(8, shape=(), dtype=int32)
            tf.Tensor(9, shape=(), dtype=int32)'''

    dataset = dataset.repeat(3).batch(7)
    for item in dataset:
        print(item)
        '''tf.Tensor([0 1 2 3 4 5 6], shape=(7,), dtype=int64)
            tf.Tensor([7 8 9 0 1 2 3], shape=(7,), dtype=int64)
            tf.Tensor([4 5 6 7 8 9 0], shape=(7,), dtype=int64)
            tf.Tensor([1 2 3 4 5 6 7], shape=(7,), dtype=int64)
            tf.Tensor([8 9], shape=(2,), dtype=int64)'''

    dataset = dataset.map(lambda x: x * 2)
    for item in dataset:
        print(item)
        '''tf.Tensor([ 0  2  4  6  8 10 12], shape=(7,), dtype=int32)
            tf.Tensor([14 16 18  0  2  4  6], shape=(7,), dtype=int32)
            tf.Tensor([ 8 10 12 14 16 18  0], shape=(7,), dtype=int32)
            tf.Tensor([ 2  4  6  8 10 12 14], shape=(7,), dtype=int32)
            tf.Tensor([16 18], shape=(2,), dtype=int32)'''

    dataset = dataset.apply(tf.data.experimental.unbatch())
    for item in dataset:
        print(item)
        '''tf.Tensor(0, shape=(), dtype=int32)
            tf.Tensor(2, shape=(), dtype=int32)
            tf.Tensor(4, shape=(), dtype=int32)
            tf.Tensor(6, shape=(), dtype=int32)
            tf.Tensor(8, shape=(), dtype=int32)
            tf.Tensor(10, shape=(), dtype=int32)
            tf.Tensor(12, shape=(), dtype=int32)
            tf.Tensor(14, shape=(), dtype=int32)
            tf.Tensor(16, shape=(), dtype=int32)
            tf.Tensor(18, shape=(), dtype=int32)
            ...*3 '''

    dataset = dataset.filter(lambda x: x < 10)  # keep only items < 10
    for item in dataset:
        print(item)
        '''tf.Tensor(0, shape=(), dtype=int32)
            tf.Tensor(2, shape=(), dtype=int32)
            tf.Tensor(4, shape=(), dtype=int32)
            tf.Tensor(6, shape=(), dtype=int32)
            tf.Tensor(8, shape=(), dtype=int32)
            tf.Tensor(0, shape=(), dtype=int32)
            tf.Tensor(2, shape=(), dtype=int32)
            tf.Tensor(4, shape=(), dtype=int32)
            tf.Tensor(6, shape=(), dtype=int32)
            tf.Tensor(8, shape=(), dtype=int32)
            tf.Tensor(0, shape=(), dtype=int32)
            tf.Tensor(2, shape=(), dtype=int32)
            tf.Tensor(4, shape=(), dtype=int32)
            tf.Tensor(6, shape=(), dtype=int32)
            tf.Tensor(8, shape=(), dtype=int32)'''

    for item in dataset.take(3):
        print(item)
        '''tf.Tensor(0, shape=(), dtype=int32)
            tf.Tensor(2, shape=(), dtype=int32)
            tf.Tensor(4, shape=(), dtype=int32)'''

