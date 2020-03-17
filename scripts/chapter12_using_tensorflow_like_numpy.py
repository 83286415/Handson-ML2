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
CHAPTER_ID = "tensorflow"
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


if __name__ == '__main__':

    # Tensors and Opterations

    # Tensors
    tf_con = tf.constant([[1., 2., 3.], [4., 5., 6.]])  # matrix: a array in constant()
    print(tf_con.shape)  # (2, 3)
    print(tf_con.dtype)  # <dtype: 'float32'>
    print(tf_con.numpy)
    # <bound method _EagerTensorBase.numpy of
    # <tf.Tensor: id=0, shape=(2, 3), dtype=float32, numpy=array([[1., 2., 3.],[4., 5., 6.]], dtype=float32)>>

    tf_scalar = tf.constant(42)  # scalar: only an int in constant()
    print(tf_scalar.shape)  # ()
    print(tf_scalar.dtype)  # <dtype: 'int32'>
    print(tf_scalar.numpy)
    # <bound method _EagerTensorBase.numpy of <tf.Tensor: id=1, shape=(), dtype=int32, numpy=42>>

    # indexing
    tf_con_part = tf_con[:, 1:]  # index operation is also like numpy
    print(tf_con_part.numpy)
    # <bound method _EagerTensorBase.numpy of
    # <tf.Tensor: id=5, shape=(2, 2), dtype=float32, numpy=array([[2., 3.], [5., 6.]], dtype=float32)>>

    tf_con_one = tf_con[:, 1]  # returns one dimension tensor whose shape is (2,)
    print(tf_con_one.numpy)
    # <bound method _EagerTensorBase.numpy of
    # <tf.Tensor: id=9, shape=(2,), dtype=float32, numpy=array([2., 5.], dtype=float32)>>

    tf_con_end = tf_con[:, 1, tf.newaxis]  # add a axis at then end of this tensor, the result is like np.newaxis
    print(tf_con_end.numpy)
    # <bound method _EagerTensorBase.numpy of
    # <tf.Tensor: id=9, shape=(2, 1), dtype=float32, numpy=array([[2.], [5.]], dtype=float32)>>

    tf_con_bng = tf_con[tf.newaxis, :]  # add a new axis at the beginning of this tensor
    print(tf_con_bng.numpy)
    # <bound method _EagerTensorBase.numpy of
    # <tf.Tensor: id=17, shape=(1, 2, 3), dtype=float32, numpy=array([[[1., 2., 3.], [4., 5., 6.]]], dtype=float32)>>

    tf_con_mid = tf_con[:, tf.newaxis, :]
    print(tf_con_mid.numpy)
    # <bound method _EagerTensorBase.numpy of
    # <tf.Tensor: id=21, shape=(2, 1, 3), dtype=float32, numpy=array([[[1., 2., 3.]], ,[[4., 5., 6.]]], dtype=float32)>>

    # TF Operations
    tf_con_add10 = tf_con + 10
    print(tf_con_add10.numpy)
    # <bound method _EagerTensorBase.numpy of
    # <tf.Tensor: id=23, shape=(2, 3), dtype=float32, numpy=array([[11., 12., 13.], [14., 15., 16.]], dtype=float32)>>

    tf_con_square = tf.square(tf_con)
    print(tf_con_square)  # tf.Tensor([[ 1.  4.  9.] [16. 25. 36.]], shape=(2, 3), dtype=float32)

    tf_con_transpose = tf.transpose(tf_con)  # the shape (2, 3) is also transposed into (3, 2)
    print(tf_con_transpose.numpy)
    # <bound method _EagerTensorBase.numpy of
    # <tf.Tensor: id=26, shape=(3, 2), dtype=float32, numpy=array([[1., 4.],[2., 5.],[3., 6.]], dtype=float32)>>

    tf_con_matmul = tf_con @ tf.transpose(tf_con)  # the operation @ is tf.matmul(), refer to cloud note for details
    print(tf_con_matmul.numpy)
    # <bound method _EagerTensorBase.numpy of
    # <tf.Tensor: id=29, shape=(2, 2), dtype=float32, numpy=array([[14., 32.], [32., 77.]], dtype=float32)>>

    # Using keras.backend
    tf_con_keras_backend = K.square(K.transpose(tf_con)) + 10  # suggest to using tf operations directly instead of this
    print(tf_con_keras_backend.numpy)
    # <bound method _EagerTensorBase.numpy of
    # <tf.Tensor: id=34, shape=(3, 2), dtype=float32, numpy=array([[11., 26.],[14., 35.],[19., 46.]], dtype=float32)>>

    # From/To NumPy