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
from scipy.special import erfc

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

# prepare for SELU
# alpha and scale to self normalize with mean 0 and standard deviation 1
alpha_0_1 = -np.sqrt(2 / np.pi) / (erfc(1 / np.sqrt(2)) * np.exp(1 / 2) - 1)
scale_0_1 = (1 - erfc(1 / np.sqrt(2)) * np.sqrt(np.e)) * np.sqrt(2 * np.pi) * (
        2 * erfc(np.sqrt(2)) * np.e ** 2 + np.pi * erfc(1 / np.sqrt(2)) ** 2 * np.e - 2 * (2 + np.pi) * erfc(
    1 / np.sqrt(2)) * np.sqrt(np.e) + np.pi + 2) ** (-1 / 2)


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


def leaky_relu(z, alpha=0.01):  # the slope of activation line, alpha in (0.01, 0.3)
    return np.maximum(alpha*z, z)


def elu(z, alpha=1):
    return np.where(z < 0, alpha * (np.exp(z) - 1), z)  # refer to the definition of this elu activation function


def selu(z, scale=scale_0_1, alpha=alpha_0_1):
    return scale * elu(z, alpha)


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

    # 1. Leaky ReLU
    # plot
    z = np.linspace(-5, 5, 200)
    plt.plot(z, leaky_relu(z, 0.05), "b-", linewidth=2)
    plt.plot([-5, 5], [0, 0], 'k-')
    plt.plot([0, 0], [-0.5, 4.2], 'k-')
    plt.grid(True)
    props = dict(facecolor='black', shrink=0.1)
    plt.annotate('Leak', xytext=(-3.5, 0.5), xy=(-5, -0.2), arrowprops=props, fontsize=14, ha="center")
    plt.title("Leaky ReLU activation function", fontsize=14)
    plt.axis([-5, 5, -0.5, 4.2])

    save_fig("leaky_relu_plot")
    # plt.show()

    # to show all activation functions in Python36\Lib\site-packages\tensorflow\python\keras\activations\__init__.py
    print([m for m in dir(keras.activations) if not m.startswith("_")])
    '''
        ['deserialize', 'elu', 'exponential', 'get', 'hard_sigmoid', 'linear', 'relu', 'selu', 'serialize', 'sigmoid', 
        'softmax', 'softplus', 'softsign', 'tanh']'''

    # to show Advanced activations in Python36\Lib\site-packages\tensorflow\python\keras\layers\__init__.py
    print([m for m in dir(keras.layers) if "relu" in m.lower()])
    '''
        ['LeakyReLU', 'PReLU', 'ReLU', 'ThresholdedReLU']'''

    # build a model with leaky ReLU
    (X_train_full, y_train_full), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
    X_train_full = X_train_full / 255.0
    X_test = X_test / 255.0
    X_valid, X_train = X_train_full[:5000], X_train_full[5000:]
    y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

    tf.random.set_random_seed(42)
    np.random.seed(42)

    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[28, 28]),
        keras.layers.Dense(300, kernel_initializer="he_normal"),  # due to # initializer above, use "he_" for ReLU
        keras.layers.LeakyReLU(),  # create a ReLU layer and add it into this model. Not use it in Dense layer as a para
        keras.layers.Dense(100, kernel_initializer="he_normal"),
        keras.layers.LeakyReLU(alpha=0.2),  # define the alpha in (0.01, 0.3) default value is 0.3
        keras.layers.Dense(10, activation="softmax")
    ])

    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=keras.optimizers.SGD(lr=1e-3),
                  metrics=["accuracy"])

    LeakyReLU_history = model.fit(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid))

    # 2. PReLU - parametric leaky ReLU
    tf.random.set_random_seed(42)
    np.random.seed(42)

    # build a model with PReLU
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[28, 28]),
        keras.layers.Dense(300, kernel_initializer="he_normal"),  # refer to "code" item in this section in cloud note
        keras.layers.PReLU(),  # # the same as leaky ReLU
        keras.layers.Dense(100, kernel_initializer="he_normal"),
        keras.layers.PReLU(),
        keras.layers.Dense(10, activation="softmax")
    ])

    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=keras.optimizers.SGD(lr=1e-3),
                  metrics=["accuracy"])  # the same as leaky ReLU

    PReLU_history = model.fit(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid))

    # 3. ELU - exponential linear unit
    keras.layers.Dense(10, activation="elu")  # as same as the sigmoid, softmax defined in layers

    # 4. SELU - self-normalized exponential linear unit
    # plot
    plt.plot(z, selu(z), "b-", linewidth=2)
    plt.plot([-5, 5], [0, 0], 'k-')
    plt.plot([-5, 5], [-1.758, -1.758], 'k--')
    plt.plot([0, 0], [-2.2, 3.2], 'k-')
    plt.grid(True)
    plt.title("SELU activation function", fontsize=14)
    plt.axis([-5, 5, -2.2, 3.2])

    save_fig("selu_plot")
    plt.show()

    # to show 1000 layers deep neural network still could matrix's mean to 0 and stds to 1 without exploding/vanishingG
    np.random.seed(42)
    Z = np.random.normal(size=(500, 100))  # standardized inputs
    for layer in range(1000):
        W = np.random.normal(size=(100, 100), scale=np.sqrt(1 / 100))  # LeCun initialization
        Z = selu(np.dot(Z, W))
        means = np.mean(Z, axis=0).mean()
        stds = np.std(Z, axis=0).mean()
        if layer % 100 == 0:
            print("Layer {}: mean {:.2f}, std deviation {:.2f}".format(layer, means, stds))

    # usage: the same as ELU. refer to "code: # 4" in this section of cloud note
    keras.layers.Dense(10, activation="selu", kernel_initializer="lecun_normal")  # lecun_ is used for SELU

    # build a model with SELU
    np.random.seed(42)
    tf.random.set_random_seed(42)

    model = keras.models.Sequential()  # must use Sequential API to create model for SELU
    model.add(keras.layers.Flatten(input_shape=[28, 28]))
    model.add(keras.layers.Dense(300, activation="selu",
                                 kernel_initializer="lecun_normal"))  # must lecun initializer
    for layer in range(99):  # won't exploding/vanishing gradient no matter how deep the network is!
        model.add(keras.layers.Dense(100, activation="selu",
                                     kernel_initializer="lecun_normal"))
    model.add(keras.layers.Dense(10, activation="softmax"))

    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=keras.optimizers.SGD(lr=1e-3),
                  metrics=["accuracy"])

    pixel_means = X_train.mean(axis=0, keepdims=True)  # data input should be scaled first.
    pixel_stds = X_train.std(axis=0, keepdims=True)
    X_train_scaled = (X_train - pixel_means) / pixel_stds
    X_valid_scaled = (X_valid - pixel_means) / pixel_stds
    X_test_scaled = (X_test - pixel_means) / pixel_stds

    history = model.fit(X_train_scaled, y_train, epochs=5,
                        validation_data=(X_valid_scaled, y_valid))