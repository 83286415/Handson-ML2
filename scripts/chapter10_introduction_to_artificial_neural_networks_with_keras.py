# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

# TensorFlow ≥2.0 is required
import tensorflow as tf
assert tf.__version__ >= "2.0"

# Common imports
import numpy as np
import os
from matplotlib.colors import ListedColormap

# chapter imports
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron

# to make this notebook's output stable across runs
np.random.seed(42)

# To plot pretty figures
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "ann"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def relu(z):
    return np.maximum(0, z)


def derivative(f, z, eps=0.000001):  # math: dao shu
    return (f(z + eps) - f(z - eps))/(2 * eps)


def heaviside(z):
    return (z >= 0).astype(z.dtype)


def mlp_xor(x1, x2, activation=heaviside):
    return activation(-activation(x1 + x2 - 1.5) + activation(x1 + x2 - 0.5) - 0.5)


# Ignore useless warnings (see SciPy issue #5998)
import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")

if __name__ == '__main__':

    # Perceptrons

    # data set
    iris = load_iris()
    X = iris.data[:, (2, 3)]  # petal length, petal width
    y = (iris.target == 0).astype(np.int)  # choose the first kind of iris

    # build model
    per_clf = Perceptron(max_iter=1000, tol=1e-3, random_state=42)
    per_clf.fit(X, y)

    y_pred = per_clf.predict([[2, 0.5]])
    print(y_pred)  # [1]

    # plot
    a = -per_clf.coef_[0][0] / per_clf.coef_[0][1]  # y=ax+b
    b = -per_clf.intercept_ / per_clf.coef_[0][1]

    axes = [0, 5, 0, 2]

    x0, x1 = np.meshgrid(
        np.linspace(axes[0], axes[1], 500).reshape(-1, 1),
        np.linspace(axes[2], axes[3], 200).reshape(-1, 1),
    )
    X_new = np.c_[x0.ravel(), x1.ravel()]
    y_predict = per_clf.predict(X_new)
    zz = y_predict.reshape(x0.shape)

    plt.figure(figsize=(10, 4))
    plt.plot(X[y == 0, 0], X[y == 0, 1], "bs", label="Not Iris-Setosa")
    plt.plot(X[y == 1, 0], X[y == 1, 1], "yo", label="Iris-Setosa")

    plt.plot([axes[0], axes[1]], [a * axes[0] + b, a * axes[1] + b], "k-", linewidth=3)
    custom_cmap = ListedColormap(['#9898ff', '#fafab0'])

    plt.contourf(x0, x1, zz, cmap=custom_cmap)
    plt.xlabel("Petal length", fontsize=14)
    plt.ylabel("Petal width", fontsize=14)
    plt.legend(loc="lower right", fontsize=14)
    plt.axis(axes)

    save_fig("perceptron_iris_plot")
    # plt.show()

    # Activation functions

    # plot Activation functions: step, sigmoid, tan, ReLU
    z = np.linspace(-5, 5, 200)

    plt.figure(figsize=(11, 4))
    plt.subplot(121)
    plt.plot(z, np.sign(z), "r-", linewidth=1, label="Step")
    plt.plot(z, sigmoid(z), "g--", linewidth=2, label="Sigmoid")
    plt.plot(z, np.tanh(z), "b-", linewidth=2, label="Tanh")
    plt.plot(z, relu(z), "m-.", linewidth=2, label="ReLU")
    plt.grid(True)
    plt.legend(loc="center right", fontsize=14)
    plt.title("Activation functions", fontsize=14)
    plt.axis([-5, 5, -1.2, 1.2])

    plt.subplot(122)
    plt.plot(z, derivative(np.sign, z), "r-", linewidth=1, label="Step")  # plot dao shu
    plt.plot(0, 0, "ro", markersize=5)
    plt.plot(0, 0, "rx", markersize=10)
    plt.plot(z, derivative(sigmoid, z), "g--", linewidth=2, label="Sigmoid")
    plt.plot(z, derivative(np.tanh, z), "b-", linewidth=2, label="Tanh")
    plt.plot(z, derivative(relu, z), "m-.", linewidth=2, label="ReLU")
    plt.grid(True)
    # plt.legend(loc="center right", fontsize=14)
    plt.title("Derivatives", fontsize=14)
    plt.axis([-5, 5, -0.2, 1.2])

    save_fig("activation_functions_plot_1")
    # plt.show()

    # plot Activation functions: heaviside, sigmoid
    x1s = np.linspace(-0.2, 1.2, 100)
    x2s = np.linspace(-0.2, 1.2, 100)
    x1, x2 = np.meshgrid(x1s, x2s)

    z1 = mlp_xor(x1, x2, activation=heaviside)
    z2 = mlp_xor(x1, x2, activation=sigmoid)

    plt.figure(figsize=(10, 4))

    plt.subplot(121)
    plt.contourf(x1, x2, z1)
    plt.plot([0, 1], [0, 1], "gs", markersize=20)
    plt.plot([0, 1], [1, 0], "y^", markersize=20)
    plt.title("Activation function: heaviside", fontsize=14)
    plt.grid(True)

    plt.subplot(122)
    plt.contourf(x1, x2, z2)
    plt.plot([0, 1], [0, 1], "gs", markersize=20)
    plt.plot([0, 1], [1, 0], "y^", markersize=20)
    plt.title("Activation function: sigmoid", fontsize=14)
    plt.grid(True)

    save_fig("activation_functions_plot_2")
    plt.show()