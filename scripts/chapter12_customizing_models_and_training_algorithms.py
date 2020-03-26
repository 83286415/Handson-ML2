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
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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


def huber_fn(y_true, y_pred):  # return a squared loss or a linear loss, which depends on the condition is_smaller_error
    error = y_true - y_pred
    is_small_error = tf.abs(error) < 1
    squared_loss = tf.square(error) / 2
    linear_loss = tf.abs(error) - 0.5
    return tf.where(is_small_error, squared_loss, linear_loss)  # where(condition, x, y) if condition true, return x.
    # tf.where: http://www.bubuko.com/infodetail-2906385.html?__cf_chl_jschl_tk__=6c2191f059d0cc320aef1b60c3e86334372bc5
    # b5-1584694162-0-AZSkNGdAz0O5WaNc0a6UFBR2s2oYMxGTOQsrXb-IANWOcdFkZ-i83hCtYDztNzcGBoKKKqLFi4ZdgQjNTjIspMk_ILqScIV9Ak
    # aPzV_Srb90SS1bWktWe2Kq1d1geiWZSBTAekp6OcLX5LA5ds4zBOcn4n3gLVOAeJc4Euq6Fi9BnqpO99jNK0EZe49CB7Z2Nrn7su3Agtc2BILcNztE
    # Gll5ShuC2BUJdujMy_PNFu-2UZzKpNWlJ9I1NwCg5adBkaQMWHlJ2aethRZow6bH05BOY2uxeDSB16XDq1vLem93_9RGBZfFLgyyD9Rfful-TQ


def create_huber(threshold=1.0):  #
    def huber_fn(y_true, y_pred):
        error = y_true - y_pred
        is_small_error = tf.abs(error) < threshold
        squared_loss = tf.square(error) / 2
        linear_loss  = threshold * tf.abs(error) - threshold**2 / 2
        return tf.where(is_small_error, squared_loss, linear_loss)
    return huber_fn


class HuberLoss(keras.losses.Loss):
    def __init__(self, threshold=1.0, **kwargs):  # input the new param and the other params from parent class
        self.threshold = threshold
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):  # the def huber_fun()
        error = y_true - y_pred
        is_small_error = tf.abs(error) < self.threshold
        squared_loss = tf.square(error) / 2
        linear_loss = self.threshold * tf.abs(error) - self.threshold**2 / 2
        return tf.where(is_small_error, squared_loss, linear_loss)

    def get_config(self):  # when saving models, get_config() is called to save configs as JSON into h5 files.
        base_config = super().get_config()  # get the dict from the parent's which mapping {params: values}
        return {**base_config, "threshold": self.threshold}  # add this new param


# customized activation function: equivalent to keras.activations.softplus() or tf.nn.softplus()
def my_softplus(z):  # return value is just tf.nn.softplus(z)
    return tf.math.log(tf.exp(z) + 1.0)


# customized initializer: equivalent to keras.initializers.glorot_normal()
def my_glorot_initializer(shape, dtype=tf.float32):
    stddev = tf.sqrt(2. / (shape[0] + shape[1]))
    return tf.random.normal(shape, stddev=stddev, dtype=dtype)


# customized regularizer: equivalent to keras.regularizers.l1(0.01)
def my_l1_regularizer(weights):
    return tf.reduce_sum(tf.abs(0.01 * weights))


# customized constraint function: equivalent to keras.constraints.nonneg() or tf.nn.relu()
def my_positive_weights(weights):  # return value is just tf.nn.relu(weights)
    return tf.where(weights < 0., tf.zeros_like(weights), weights)


class MyL1Regularizer(keras.regularizers.Regularizer):
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, weights):
        return tf.reduce_sum(tf.abs(self.factor * weights))

    def get_config(self):
        return {"factor": self.factor}


class HuberMetric(keras.metrics.Metric):
    def __init__(self, threshold=1.0, **kwargs):
        super().__init__(**kwargs) # handles base args (e.g., dtype)
        self.threshold = threshold
        # self.huber_fn = create_huber(threshold) # TODO: investigate why this fails
        self.total = self.add_weight("total", initializer="zeros")
        self.count = self.add_weight("count", initializer="zeros")

    def huber_fn(self, y_true, y_pred): # workaround
        error = y_true - y_pred
        is_small_error = tf.abs(error) < self.threshold
        squared_loss = tf.square(error) / 2
        linear_loss = self.threshold * tf.abs(error) - self.threshold**2 / 2
        return tf.where(is_small_error, squared_loss, linear_loss)

    def update_state(self, y_true, y_pred, sample_weight=None):
        metric = self.huber_fn(y_true, y_pred)
        self.total.assign_add(tf.reduce_sum(metric))
        self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))

    def result(self):
        return self.total / self.count

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "threshold": self.threshold}


class HuberMetricSimple(keras.metrics.Mean):  # a simple version of HuberMetric
    def __init__(self, threshold=1.0, name='HuberMetric', dtype=None):
        self.threshold = threshold
        self.huber_fn = create_huber(threshold)
        super().__init__(name=name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        metric = self.huber_fn(y_true, y_pred)
        super(HuberMetric, self).update_state(metric, sample_weight)

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "threshold": self.threshold}


# single input layer class
class MyDense(keras.layers.Layer):  # refer to "Custom Layer" in my cloud note
    def __init__(self, units, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)

    def build(self, batch_input_shape):
        self.kernel = self.add_weight(
            name="kernel", shape=[batch_input_shape[-1], self.units],
            initializer="glorot_normal")
        self.bias = self.add_weight(
            name="bias", shape=[self.units], initializer="zeros")
        super().build(batch_input_shape) # must be at the end

    def call(self, X):
        return self.activation(X @ self.kernel + self.bias)

    def compute_output_shape(self, batch_input_shape):
        return tf.TensorShape(batch_input_shape.as_list()[:-1] + [self.units])

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "units": self.units,
                "activation": keras.activations.serialize(self.activation)}


# multi output layer class
class MyMultiLayer(keras.layers.Layer):  # refer to "Custom Layer" in my cloud note

    def call(self, X):
        X1, X2 = X
        return X1 + X2, X1 * X2

    def compute_output_shape(self, batch_input_shape):
        batch_input_shape1, batch_input_shape2 = batch_input_shape
        return [batch_input_shape1, batch_input_shape2]


# create a layer class with a different behavior during training and testing
class AddGaussianNoise(keras.layers.Layer):  # refer to "Custom Layer" in my cloud note
    def __init__(self, stddev, **kwargs):
        super().__init__(**kwargs)
        self.stddev = stddev

    def call(self, X, training=None):
        if training:
            noise = tf.random.normal(tf.shape(X), stddev=self.stddev)
            return X + noise  # only add noise in training process
        else:
            return X

    def compute_output_shape(self, batch_input_shape):
        return batch_input_shape


# Custom Models
class ResidualBlock(keras.layers.Layer):
    def __init__(self, n_layers, n_neurons, **kwargs):
        super().__init__(**kwargs)
        self.n_layers = n_layers  # not shown in the book
        self.n_neurons = n_neurons  # not shown
        self.hidden = [keras.layers.Dense(n_neurons, activation="elu",
                                          kernel_initializer="he_normal")
                       for _ in range(n_layers)]

    def call(self, inputs):
        Z = inputs
        for layer in self.hidden:
            Z = layer(Z)
        return inputs + Z

    def get_config(self):  # to support save() and load() function
        base_config = super().get_config()
        return {**base_config, "n_layers": self.n_layers, "n_neurons": self.n_neurons}


class ResidualRegressor(keras.models.Model):
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim                                 # not shown in the book
        self.hidden1 = keras.layers.Dense(30, activation="elu",
                                          kernel_initializer="he_normal")
        self.block1 = ResidualBlock(2, 30)
        self.block2 = ResidualBlock(2, 30)
        self.out = keras.layers.Dense(output_dim)

    def call(self, inputs):
        Z = self.hidden1(inputs)
        for _ in range(1 + 3):
            Z = self.block1(Z)
        Z = self.block2(Z)
        return self.out(Z)

    def get_config(self):  # to support save() and load() function
        base_config = super().get_config()
        return {**base_config,
                "output_dim": self.output_dim}


# Losses and Metrics Based on Model Internals
class ReconstructingRegressor(keras.models.Model):
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.hidden = [keras.layers.Dense(30, activation="selu",
                                          kernel_initializer="lecun_normal")
                       for _ in range(5)]
        self.out = keras.layers.Dense(output_dim)
        # TODO: check https://github.com/tensorflow/tensorflow/issues/26260
        #self.reconstruction_mean = keras.metrics.Mean(name="reconstruction_error")

    def build(self, batch_input_shape):
        n_inputs = batch_input_shape[-1]
        self.reconstruct = keras.layers.Dense(n_inputs)
        super().build(batch_input_shape)

    def call(self, inputs, training=None):
        Z = inputs
        for layer in self.hidden:
            Z = layer(Z)
        reconstruction = self.reconstruct(Z)
        recon_loss = tf.reduce_mean(tf.square(reconstruction - inputs))
        self.add_loss(0.05 * recon_loss)
        #if training:
        #    result = self.reconstruction_mean(recon_loss)
        #    self.add_metric(result)
        return self.out(Z)


if __name__ == '__main__':

    # Custom Loss Functions

    # data set
    housing = fetch_california_housing()
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        housing.data, housing.target.reshape(-1, 1), random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_full, y_train_full, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    X_test_scaled = scaler.transform(X_test)

    # plot
    plt.figure(figsize=(8, 3.5))
    z = np.linspace(-4, 4, 200)
    plt.plot(z, huber_fn(0, z), "b-", linewidth=2, label="huber($z$)")
    plt.plot(z, z ** 2 / 2, "b:", linewidth=1, label=r"$\frac{1}{2}z^2$")
    plt.plot([-1, -1], [0, huber_fn(0., -1.)], "r--")
    plt.plot([1, 1], [0, huber_fn(0., 1.)], "r--")
    plt.gca().axhline(y=0, color='k')
    plt.gca().axvline(x=0, color='k')
    plt.axis([-4, 4, 0, 4])
    plt.grid(True)
    plt.xlabel("$z$")
    plt.legend(fontsize=14)
    plt.title("Huber loss", fontsize=14)
    save_fig('huber_loss', tight_layout=False)
    # plt.show()

    input_shape = X_train.shape[1:]
    print(X_train.shape)  # (11610, 8)  # 11610 samples, each sample gets 8 features
    print(input_shape)  # (8,)  # limit the input shape to make sure one sample input a time

    model = keras.models.Sequential([
        keras.layers.Dense(30, activation="selu", kernel_initializer="lecun_normal",
                           input_shape=input_shape),
        keras.layers.Dense(1),
    ])

    model.compile(loss=huber_fn, optimizer="nadam", metrics=["mae"])
    model.fit(X_train_scaled, y_train, epochs=2,
              validation_data=(X_valid_scaled, y_valid))

    # Saving and Loading Models That Contain Custom Components

    # save the model last session defined
    save_model(model, "my_model_with_a_custom_loss")

    # load
    model = load_model("my_model_with_a_custom_loss.h5",
                       custom_objects={"huber_fn": huber_fn})  # but cannot custom threshold in huber_fn()

    # define another huber function to custom threshold input
    model.compile(loss=create_huber(2.0), optimizer="nadam", metrics=["mae"])  # now we can custom threshold in huber
    model.fit(X_train_scaled, y_train, epochs=2, validation_data=(X_valid_scaled, y_valid))

    # save
    save_model(model, "my_model_with_a_custom_loss_threshold_2")

    # load
    model = load_model("my_model_with_a_custom_loss_threshold_2.h5",
                       custom_objects={"huber_fn": create_huber(2.0)})  # but cannot save param's value in model
    model.fit(X_train_scaled, y_train, epochs=2, validation_data=(X_valid_scaled, y_valid))

    # define a class to hold threshold config when saving a model
    model = keras.models.Sequential([
        keras.layers.Dense(30, activation="selu", kernel_initializer="lecun_normal",
                           input_shape=input_shape),
        keras.layers.Dense(1),
    ])

    model.compile(loss=HuberLoss(2.), optimizer="nadam", metrics=["mae"])  # add loss class with threshold param
    model.fit(X_train_scaled, y_train, epochs=2, validation_data=(X_valid_scaled, y_valid))

    # save
    save_model(model, "my_model_with_a_custom_loss_class")  # save the model with loss threshold

    # load (commit it out for now this bug is not fixed in Keras yet. refer to RP25956)
    # model = load_model("my_model_with_a_custom_loss_class.h5",
    #                    custom_objects={"HuberLoss": HuberLoss})
    # print(model.loss.threshold)   # now we can retrieve params value after load the model

    # Other Custom Functions

    layer = keras.layers.Dense(1, activation=my_softplus,
                               kernel_initializer=my_glorot_initializer,
                               kernel_regularizer=my_l1_regularizer,
                               kernel_constraint=my_positive_weights)  # all components are customized

    model = keras.models.Sequential([
        keras.layers.Dense(30, activation="selu", kernel_initializer="lecun_normal",
                           input_shape=input_shape),
        keras.layers.Dense(1, activation=my_softplus,
                           kernel_regularizer=my_l1_regularizer,
                           kernel_constraint=my_positive_weights,
                           kernel_initializer=my_glorot_initializer),
    ])

    model.compile(loss="mse", optimizer="nadam", metrics=["mae"])
    model.fit(X_train_scaled, y_train, epochs=2, validation_data=(X_valid_scaled, y_valid))

    # save and load
    save_model(model, "my_model_with_many_custom_parts.h5")
    model = load_model(
                        "my_model_with_many_custom_parts.h5",
                        custom_objects={
                            "my_l1_regularizer": my_l1_regularizer,
                            "my_positive_weights": lambda: my_positive_weights,  # tf.wherer returns a tf object
                            "my_glorot_initializer": my_glorot_initializer,
                            "my_softplus": my_softplus,
                        })

    model = keras.models.Sequential([
        keras.layers.Dense(30, activation="selu", kernel_initializer="lecun_normal",
                           input_shape=input_shape),
        keras.layers.Dense(1, activation=my_softplus,
                           kernel_regularizer=MyL1Regularizer(0.01),  # customized regularizer class with param input
                           kernel_constraint=my_positive_weights,
                           kernel_initializer=my_glorot_initializer),
    ])

    model.compile(loss="mse", optimizer="nadam", metrics=["mae"])
    model.fit(X_train_scaled, y_train, epochs=2, validation_data=(X_valid_scaled, y_valid))

    # save and load
    save_model(model, "my_model_with_many_custom_parts.h5")
    model = load_model(
                        "my_model_with_many_custom_parts.h5",
                        custom_objects={
                            "MyL1Regularizer": MyL1Regularizer,
                            "my_positive_weights": lambda: my_positive_weights,
                            "my_glorot_initializer": my_glorot_initializer,
                            "my_softplus": my_softplus,
                        })

    # Custom Metrics

    precision = keras.metrics.Precision()
    precision([0, 1, 1, 1, 0, 1, 0, 1], [1, 1, 0, 1, 0, 1, 0, 1])
    print(precision([0, 1, 0, 0, 1, 0, 1, 1], [1, 0, 1, 1, 0, 0, 0, 0]))  # tf.Tensor(0.5, shape=(), dtype=float32)
    print(precision.result())  # tf.Tensor(0.5, shape=(), dtype=float32)
    print(precision.variables)
    # [<tf.Variable 'true_positives:0' shape=(1,) dtype=float32, numpy=array([4.], dtype=float32)>,
    # <tf.Variable 'false_positives:0' shape=(1,) dtype=float32, numpy=array([4.], dtype=float32)>]
    precision.reset_states()

    # Streaming Metrics
    m = HuberMetric(2.)

    # m test:
    # total = 2 * |10 - 2| - 2²/2 = 14
    # count = 1
    # result = 14 / 1 = 14
    # print(m(tf.constant([[2.]]), tf.constant([[10.]])))

    # m test:
    # total = total + (|1 - 0|² / 2) + (2 * |9.25 - 5| - 2² / 2) = 14 + 7 = 21
    # count = count + 2 = 3
    # result = total / count = 21 / 3 = 7
    m(tf.constant([[0.], [5.]]), tf.constant([[1.], [9.25]]))
    print(m.result())
    print(m.variables)
    m.reset_states()  # reset to check the variables list
    print(m.variables)

    # build a model
    model = keras.models.Sequential([
        keras.layers.Dense(30, activation="selu", kernel_initializer="lecun_normal", input_shape=input_shape),
        keras.layers.Dense(1),
    ])
    model.compile(loss=create_huber(2.0), optimizer="nadam", metrics=[HuberMetric(2.0)])
    model.fit(X_train_scaled.astype(np.float32), y_train.astype(np.float32), epochs=2)

    save_model(model, "my_model_with_a_custom_metric.h5")
    # model.fit(X_train_scaled.astype(np.float32), y_train.astype(np.float32), epochs=2)
    print(model.metrics[0].threshold)  # 2.0

    # Custom Layers

    # activation="exponential" also can be like this layer:
    exponential_layer = keras.layers.Lambda(lambda x: tf.exp(x))

    model = keras.models.Sequential([
        keras.layers.Dense(30, activation="relu", input_shape=input_shape),
        keras.layers.Dense(1),  # activation="exponential" in this layer can replace exponential_layer below
        exponential_layer
    ])
    model.compile(loss="mse", optimizer="nadam")
    model.fit(X_train_scaled, y_train, epochs=5,
              validation_data=(X_valid_scaled, y_valid))
    model.evaluate(X_test_scaled, y_test)

    # custom single input layer
    model = keras.models.Sequential([
        MyDense(30, activation="relu", input_shape=input_shape),
        MyDense(1)
    ])

    model.compile(loss="mse", optimizer="nadam")
    model.fit(X_train_scaled, y_train, epochs=2,
              validation_data=(X_valid_scaled, y_valid))
    model.evaluate(X_test_scaled, y_test)

    save_model(model, "my_model_with_a_custom_layer.h5")

    # custom multi input layer
    inputs1 = keras.layers.Input(shape=[2])
    inputs2 = keras.layers.Input(shape=[2])
    outputs1, outputs2 = MyMultiLayer()((inputs1, inputs2))

    model.compile(loss="mse", optimizer="nadam")  # i think it's NOT correct here
    model.fit(X_train_scaled, y_train, epochs=2,
              validation_data=(X_valid_scaled, y_valid))
    model.evaluate(X_test_scaled, y_test)

    # Custom Models

    # data set
    X_new_scaled = X_test_scaled

    model = ResidualRegressor(1)
    model.compile(loss="mse", optimizer="nadam")
    history = model.fit(X_train_scaled, y_train, epochs=5)
    score = model.evaluate(X_test_scaled, y_test)
    y_pred = model.predict(X_new_scaled)

    # save_model(model, "my_custom_model.ckpt")  # not supported for now
    # model = keras.models.load_model("my_custom_model.ckpt")

    # Losses and Metrics Based on Model Internals

    model = ReconstructingRegressor(1)
    model.compile(loss="mse", optimizer="nadam")
    history_reconstruction = model.fit(X_train_scaled, y_train, epochs=2)
    y_pred_reconstruction = model.predict(X_test_scaled)

    # Computing Gradients with Autodiff

