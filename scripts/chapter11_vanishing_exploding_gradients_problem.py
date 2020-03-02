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


if __name__ == '__main__':

    # Xavier and He Initialization

    print('Xavier and He Initialization')
    # to show all initializers in keras
    print([name for name in dir(keras.initializers) if not name.startswith("_")])
    '''
        ['Constant', 'GlorotNormal', 'GlorotUniform', 'Identity', 'Initializer', 'Ones', 'Orthogonal', 'RandomNormal', 
        'RandomUniform', 'TruncatedNormal', 'VarianceScaling', 'Zeros', 'constant', 'deserialize', 'get', 
        'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform', 'identity', 'lecun_normal', 'lecun_uniform', 
        'ones', 'orthogonal', 'serialize', 'zeros']
    '''

    # change kernel_init to he_normal
    keras.layers.Dense(10, activation="relu", kernel_initializer="he_normal")  # kernel_init default: GlorotUniform

    # change init's mode from fan_in into fan_avg
    he_avg_init = keras.initializers.VarianceScaling(scale=2., mode='fan_avg', distribution='uniform')  # default:fan_in
    keras.layers.Dense(10, activation="relu", kernel_initializer=he_avg_init)

    # Nonsaturating Activation Functions

    print('Nonsaturating Activation Functions')
