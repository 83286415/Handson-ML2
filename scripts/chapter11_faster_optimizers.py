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

    optimizer = keras.optimizers.SGD(lr=0.001, momentum=0.9)  # It's easy to add momentum optimizer. Just add this para.

    # Nesterov Accelerated Gradient