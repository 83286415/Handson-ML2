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


if __name__ == '__main__':

    # Building an Image Classifier Using the Sequential API

    # load data - Fashion MNIST
    fashion_mnist = keras.datasets.fashion_mnist
    (X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()  # total 70000 samples

    print(X_train_full.shape)  # (60000, 28, 28)
    print(X_test.shape)  # (10000, 28, 28)
    print(X_train_full.dtype)  # uint8

    # validation set and train set
    X_valid, X_train = X_train_full[:5000] / 255., X_train_full[5000:] / 255.  # divided by 255 to scale pixel into 0~1
    y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

    X_test = X_test / 255.
    print(X_train.shape)  # (55000, 28, 28)

    class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                   "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]  # a list of y label's mean

    # plot
    n_rows = 4
    n_cols = 10
    plt.figure(figsize=(n_cols * 1.2, n_rows * 1.2))
    for row in range(n_rows):  # row starts from 0 to 9 , which is included
        for col in range(n_cols):
            index = n_cols * row + col
            plt.subplot(n_rows, n_cols, index + 1)
            plt.imshow(X_train[index], cmap="binary", interpolation="nearest")
            plt.axis('off')
            plt.title(class_names[y_train[index]], fontsize=12)
    plt.subplots_adjust(wspace=0.2, hspace=0.5)
    save_fig('fashion_mnist_plot', tight_layout=False)
    # plt.show()

    # build model
    model = keras.models.Sequential()  # create a simplest keras model
    # model.add(keras.layers.InpuLayer(input_shape=[28, 28]))  # use this one or flatten below as the input layer
    model.add(keras.layers.Flatten(input_shape=[28, 28]))  # preprocessing data set: reshape(-1, 1)
    model.add(keras.layers.Dense(300, activation="relu"))  # 1. weight matrix between input and neurons in this layer
    model.add(keras.layers.Dense(100, activation="relu"))  # 2. as well as the bias vector of each neuron of this layer
    model.add(keras.layers.Dense(10, activation="softmax"))  # 10 kinds of labels so 10 output neuron with softmax
    # layers' param: kernel_initializer, bias_initializer and other initializer refer to https://keras.io/initializers/

    keras.backend.clear_session()  # TODO: backend Not found
    np.random.seed(42)
    tf.random.set_random_seed(42)

    print(model.summary())
    '''
    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    flatten (Flatten)            (None, 784)               0           # 784 = 28*28
    _________________________________________________________________
    dense (Dense)                (None, 300)               235500      # 23550 = 784 * 300 + 300bias
    _________________________________________________________________
    dense_1 (Dense)              (None, 100)               30100       # 30100 = 300 * 100 + 100bias
    _________________________________________________________________
    dense_2 (Dense)              (None, 10)                1010        # 1010 = 100 * 10 + 10bias
    =================================================================
    Total params: 266,610
    Trainable params: 266,610
    Non-trainable params: 0
    _________________________________________________________________
    '''

    # plot model: maybe tensorboard is better than this plot_model()
    keras.utils.plot_model(model, to_file="D:/AI/handson-ml2-master/images/ann/my_fashion_mnist_model.png",
                           show_shapes=True)  # ? in png means "None"

    hidden1 = model.layers[1]
    print(hidden1.name)  # dense

    weights, biases = hidden1.get_weights()  # set_weights() could be used to set weights
    print(weights.shape)  # (784, 300)
    print(weights)
    '''[[ 0.06918676 -0.00970715 -0.01093315 ...  0.02926975  0.00293471
      -0.01413641]
     [ 0.01723376  0.00755721  0.03315927 ...  0.06185281  0.06487153
       0.04238527]
     [-0.0688936  -0.06460477  0.05693319 ... -0.03042184 -0.05042943
       0.00235905]
     ...
     [-0.01471603  0.03180552  0.01331522 ... -0.0113719  -0.0021135
       0.01305767]
     [-0.00393201 -0.03535463  0.05082566 ...  0.05848566 -0.0188842
      -0.03579393]
     [-0.06715128  0.03948358  0.00224125 ...  0.04806222 -0.00379574
       0.00937662]]'''
    print(biases.shape)  # (300,)
    print(biases)
    '''[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]'''

    # compile
    model.compile(loss="sparse_categorical_crossentropy",  # loss=keras.losses.sparse_categorical_crossentropy
                  optimizer="sgd",  # optimizer=keras.optimizers.SGD()
                  metrics=["accuracy"])  # metrics=[keras.metrics.sparse_categorical_accuracy]
    # https://keras.io/losses, https://keras.io/optimizers, and https://keras.io/metrics

    # fit
    history = model.fit(X_train, y_train, epochs=30,
                        validation_data=(X_valid, y_valid))  # or use  validation_split=0.1 instead of validation_data
    # fit params: class_weight, sample_weight refer to cloud note's "fit's params"
    print(history.params)
    '''{'batch_size': 32, 'epochs': 30, 'steps': 1719, 'samples': 55000, 'verbose': 0, 'do_validation': True, 
    'metrics': ['loss', 'accuracy', 'val_loss', 'val_accuracy']}'''
    print(history.epoch)
    # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
    print(history.history.keys())  # dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])

    # plot loss and accuracy
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)  # set the vertical range to [0-1]
    save_fig("keras_learning_curves_plot")
    # plt.show()

    # evaluate
    model.evaluate(X_test, y_test)  # loss: 0.2227 - accuracy: 0.8822

    # prediction
    X_new = X_test[:9]
    y_proba = model.predict(X_new)
    print(y_proba.round(2))  # save 2 digits for the result
    '''[[0.   0.   0.   0.   0.   0.   0.   0.   0.   1.  ]
         [0.   0.   1.   0.   0.   0.   0.   0.   0.   0.  ]
         [0.   1.   0.   0.   0.   0.   0.   0.   0.   0.  ]
         [0.   1.   0.   0.   0.   0.   0.   0.   0.   0.  ]
         [0.1  0.   0.02 0.   0.   0.   0.87 0.   0.   0.  ]
         [0.   1.   0.   0.   0.   0.   0.   0.   0.   0.  ]
         [0.   0.   0.01 0.   0.97 0.   0.02 0.   0.   0.  ]
         [0.   0.   0.   0.   0.   0.   1.   0.   0.   0.  ]
         [0.   0.   0.   0.   0.   0.99 0.   0.   0.   0.  ]
         [0.   0.   0.   0.   0.   0.   0.   1.   0.   0.  ]]'''

    y_pred = model.predict_classes(X_new)
    print(y_pred)  # [9 2 1 1 6 1 4 6 5 7]
    print(np.array(class_names)[y_pred])  # print the classification name according to the results
    # ['Ankle boot' 'Pullover' 'Trouser' 'Trouser' 'Shirt' 'Trouser' 'Coat' 'Shirt' 'Sandal' 'Sneaker']
    y_new = y_test[:9]

    # plot
    plt.figure(figsize=(21.6, 2.4))
    for index, image in enumerate(X_new):
        plt.subplot(1, 9, index + 1)
        plt.imshow(image, cmap="binary", interpolation="nearest")
        plt.axis('off')
        plt.title(class_names[y_test[index]], fontsize=12)
    plt.subplots_adjust(wspace=0.2, hspace=0.5)
    save_fig('fashion_mnist_images_plot', tight_layout=False)
    # plt.show()

    # Regression MLP

    # data set
    housing = fetch_california_housing()

    X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, random_state=42)  # default 0.25

    # scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)
    X_test = scaler.transform(X_test)

    # build model
    model = keras.models.Sequential([
        keras.layers.Dense(30, activation="relu", input_shape=X_train.shape[1:]),
        keras.layers.Dense(1)  # only one hidden unit cell for regression mission without any activation function
    ])
    model.compile(loss="mean_squared_error", optimizer=keras.optimizers.SGD(lr=1e-3))
    history = model.fit(X_train, y_train, epochs=20, validation_data=(X_valid, y_valid))
    mse_test = model.evaluate(X_test, y_test)
    print(mse_test)  # 0.4398445256458696
    X_new = X_test[:3]
    y_pred = model.predict(X_new)
    print(y_pred)  # [[0.75302976] [1.7815856 ] [3.1391568 ]] the outputs are prediction of house price

    # plot
    plt.plot(pd.DataFrame(history.history))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()

    # Functional API

    # op random seed
    np.random.seed(42)
    tf.random.set_random_seed(42)

    # 1. build a model like the one in cloud note "Building Complex Models Using the Functional API"
    input_ = keras.layers.Input(shape=X_train.shape[1:])
    hidden1 = keras.layers.Dense(30, activation="relu")(input_)
    hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
    concat = keras.layers.concatenate([input_, hidden2])  # two input of this layer in ONE list
    output = keras.layers.Dense(1)(concat)
    model = keras.models.Model(inputs=[input_], outputs=[output])

    print(model.summary())
    '''
        Model: "model"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_1 (InputLayer)            [(None, 8)]          0                                            
    __________________________________________________________________________________________________
    dense_2 (Dense)                 (None, 30)           270         input_1[0][0]                    
    __________________________________________________________________________________________________
    dense_3 (Dense)                 (None, 30)           930         dense_2[0][0]                    
    __________________________________________________________________________________________________
    concatenate (Concatenate)       (None, 38)           0           input_1[0][0]                    
                                                                     dense_3[0][0]                    
    __________________________________________________________________________________________________
    dense_4 (Dense)                 (None, 1)            39          concatenate[0][0]                
    ==================================================================================================
    Total params: 1,239
    Trainable params: 1,239
    Non-trainable params: 0
    __________________________________________________________________________________________________
    '''

    # compile, train, evaluate and predict
    model.compile(loss="mean_squared_error", optimizer=keras.optimizers.SGD(lr=1e-3))
    model.fit(X_train, y_train, epochs=20,
              validation_data=(X_valid, y_valid))
    mse_test = model.evaluate(X_test, y_test)
    print(mse_test)  # 0.4045557499855988
    y_pred = model.predict(X_new)
    print(y_pred)  # [[0.4686121] [1.8569269] [3.340442 ]] three output of the house price prediction

    np.random.seed(42)
    tf.random.set_random_seed(42)

    # 2. multi-input
    input_A = keras.layers.Input(shape=[5], name="wide_input")
    input_B = keras.layers.Input(shape=[6], name="deep_input")
    hidden1 = keras.layers.Dense(30, activation="relu")(input_B)
    hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
    concat = keras.layers.concatenate([input_A, hidden2])
    output = keras.layers.Dense(1, name="output")(concat)
    model = keras.models.Model(inputs=[input_A, input_B], outputs=[output])  # two inputs in ONE list or tuple

    # compile, train, evaluate and predict
    model.compile(loss="mse", optimizer=keras.optimizers.SGD(lr=1e-3))

    X_train_A, X_train_B = X_train[:, :5], X_train[:, 2:]  # split train data into two input
    X_valid_A, X_valid_B = X_valid[:, :5], X_valid[:, 2:]
    X_test_A, X_test_B = X_test[:, :5], X_test[:, 2:]
    X_new_A, X_new_B = X_test_A[:3], X_test_B[:3]  # split test data into two input

    model.fit([X_train_A, X_train_B], y_train, epochs=20,
              validation_data=((X_valid_A, X_valid_B), y_valid))  # fit input: two input in ONE tuple or list
    mse_test = model.evaluate((X_test_A, X_test_B), y_test)  # evaluate input: two input in ONE tuple or list
    print(mse_test)  # 0.4249315629633822 not better than the model above
    y_pred = model.predict((X_new_A, X_new_B))  # predict input: two input in ONE tuple or list
    print(y_pred)  # [[0.2949911] [1.9546194] [3.4345765]]

    # 3. multi-output
    np.random.seed(42)
    tf.random.set_random_seed(42)

    input_A = keras.layers.Input(shape=[5], name="wide_input")
    input_B = keras.layers.Input(shape=[6], name="deep_input")
    hidden1 = keras.layers.Dense(30, activation="relu")(input_B)
    hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
    concat = keras.layers.concatenate([input_A, hidden2])
    output = keras.layers.Dense(1, name="main_output")(concat)
    aux_output = keras.layers.Dense(1, name="aux_output")(hidden2)
    model = keras.models.Model(inputs=[input_A, input_B], outputs=[output, aux_output])  # two params

    # compile, train, evaluate and predict
    model.compile(loss=["mse", "mse"], loss_weights=[0.9, 0.1], optimizer=keras.optimizers.SGD(lr=1e-3))  # two params
    model.fit([X_train_A, X_train_B], [y_train, y_train], epochs=20,
              validation_data=([X_valid_A, X_valid_B], [y_valid, y_valid]))  # two input in ONE list or tuple
    total_loss, main_loss, aux_loss = model.evaluate([X_test_A, X_test_B], [y_test, y_test])  # three losses
    print(total_loss, main_loss, aux_loss)  # 0.4683263566142829 0.41918433 0.91364497
    y_pred_main, y_pred_aux = model.predict([X_new_A, X_new_B])  # two output lists:
    print(y_pred_main, y_pred_aux)  # three output of the house price prediction in two lists:
    # [[0.9526369] [1.9273019] [2.5151913]], [[0.25949997] [1.9820837 ] [3.3337932 ]]

    # Sbclassing API

    # the class WideAndDeepModel define above this main
    model = WideAndDeepModel(30, activation="relu")

    model.compile(loss="mse", loss_weights=[0.9, 0.1], optimizer=keras.optimizers.SGD(lr=1e-3))
    history = model.fit((X_train_A, X_train_B), (y_train, y_train), epochs=10,
                        validation_data=((X_valid_A, X_valid_B), (y_valid, y_valid)))
    total_loss, main_loss, aux_loss = model.evaluate((X_test_A, X_test_B), (y_test, y_test))
    print(total_loss, main_loss, aux_loss)
    y_pred_main, y_pred_aux = model.predict((X_new_A, X_new_B))
    print(y_pred_main, y_pred_aux)