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
import time
import contextlib

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


# used to test autodiff for computing gradient
def f(w1, w2):
    return 3 * w1 ** 2 + 2 * w1 * w2


def f_stop(w1, w2):  # stop gradients from back propagating
    return 3 * w1 ** 2 + tf.stop_gradient(2 * w1 * w2)


# to handle the large inputs problem
@tf.custom_gradient
def my_better_softplus(z):
    exp = tf.exp(z)

    def my_softplus_gradients(grad):
        return grad / (1 + 1 / exp)
    return tf.math.log(exp + 1), my_softplus_gradients


def random_batch(X, y, batch_size=32):  # return batch index randomly
    idx = np.random.randint(len(X), size=batch_size)
    return X[idx], y[idx]


def progress_bar(iteration, total, size=30):
    running = iteration < total
    c = ">" if running else "="
    p = (size - 1) * iteration // total
    fmt = "{{:-{}d}}/{{}} [{{}}]".format(len(str(total)))
    params = [iteration, total, "=" * p + c + "." * (size - p - 1)]
    return fmt.format(*params)


def print_status_bar(iteration, total, loss, metrics=None, size=30):
    metrics = " - ".join(["{}: {:.4f}".format(m.name, m.result())
                         for m in [loss] + (metrics or [])])
    end = "" if iteration < total else "\n"
    print("\r{} - {}".format(progress_bar(iteration, total), metrics), end=end)


@contextlib.contextmanager
def assert_raises(error_class):
    try:
        yield
    except error_class as e:
        print('Caught expected exception \n  {}: {}'.format(error_class, e))
    except Exception as e:
        print('Got unexpected exception \n  {}: {}'.format(type(e), e))
    else:
        raise Exception('Expected {} to be raised but no error was raised!'.format(
            error_class))


@tf.function(input_signature=[tf.TensorSpec([None, 28, 28], tf.float32)])
def shrink(images):
    print("Tracing", images)
    return images[:, ::2, ::2] # drop half the rows and columns


def cube(x):
    return x ** 3


@tf.function
def add_10(x):
    for i in range(10):
        x += 1
        print(x)
    return x


@tf.function
def add_10_condition(x):
    condition = lambda i, x: tf.less(i, 10)
    body = lambda i, x: (tf.add(i, 1), tf.add(x, 1))
    final_i, final_x = tf.while_loop(condition, body, [tf.constant(0), x])
    return final_x


# Custom loss function
def my_mse(y_true, y_pred):
    print("Tracing loss my_mse()")
    return tf.reduce_mean(tf.square(y_pred - y_true))


# Custom metric function
def my_mae(y_true, y_pred):
    print("Tracing metric my_mae()")
    return tf.reduce_mean(tf.abs(y_pred - y_true))


# Custom layer
class MyDense(keras.layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.units),
                                      initializer='uniform',
                                      trainable=True)
        self.biases = self.add_weight(name='bias',
                                      shape=(self.units,),
                                      initializer='zeros',
                                      trainable=True)
        super().build(input_shape)

    def call(self, X):
        print("Tracing MyDense.call()")
        return self.activation(X @ self.kernel + self.biases)  # @: np.dot(X, (self.kernel + self.biases))


# Custom model
class MyModel(keras.models.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # super().__init__(dynamic=True, **kwargs) to stop converting Python to TF function
        self.hidden1 = MyDense(30, activation="relu")
        self.hidden2 = MyDense(30, activation="relu")
        self.output_ = MyDense(1)

    def call(self, input):
        print("Tracing MyModel.call()")
        hidden1 = self.hidden1(input)
        hidden2 = self.hidden2(hidden1)
        concat = keras.layers.concatenate([input, hidden2])
        output = self.output_(concat)
        return output


if __name__ == '__main__':

    # Tensorflow Functions and Graphs

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

    # TensorFlow Functions
    print(cube(2))  # return 8

    print(cube(tf.constant(2.0)))  # tf.Tensor(8.0, shape=(), dtype=float32)

    tf_cube = tf.function(cube)
    print(tf_cube(2))                   # tf.Tensor(8, shape=(), dtype=int32)  return a tensor
    print(tf_cube(tf.constant(2.0)))    # tf.Tensor(8.0, shape=(), dtype=float32)     a tensor

    # TF Functions and Concrete Functions
    concrete_function = tf_cube.get_concrete_function(tf.constant(2.0))  # tf.constant type is allowed
    print(concrete_function(tf.constant(2.0)))  # tf.Tensor(8.0, shape=(), dtype=float32)
    with assert_raises(tf.errors.InvalidArgumentError):
        concrete_function(tf.string('a'))  # type error caught!

    # Exploring Function Definitions and Graphs
    print(concrete_function.graph)  # FuncGraph(name=cube, id=2050414691664)
    ops = concrete_function.graph.get_operations()
    print('operations: ', ops)
    # operations:  [<tf.Operation 'x' type=Placeholder>, <tf.Operation 'pow/y' type=Const>,
    # <tf.Operation 'pow' type=Pow>, <tf.Operation 'Identity' type=Identity>]
    pow_op = ops[2]
    print(list(pow_op.inputs))
    # [<tf.Tensor 'x:0' shape=() dtype=float32>, <tf.Tensor 'pow/y:0' shape=() dtype=float32>]
    print(pow_op.outputs)  # [<tf.Tensor 'pow:0' shape=() dtype=float32>]
    print(concrete_function.graph.get_operation_by_name('x'))
    '''
    name: "x"
    op: "Placeholder"
    attr {
      key: "_user_specified_name"
      value {
        s: "x"
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "shape"
      value {
        shape {
        }
      }
    }'''
    print(concrete_function.graph.get_tensor_by_name('Identity:0'))  # Tensor("Identity:0", shape=(), dtype=float32)
    print(concrete_function.function_def.signature)
    ''' name: "__inference_cube_16"
        input_arg {
          name: "x"
          type: DT_FLOAT
        }
        output_arg {
          name: "identity"
          type: DT_FLOAT
        }'''

    # How TF Functions Trace Python Functions to Extract Their Computation Graphs
    img_batch_1 = tf.random.uniform(shape=[100, 28, 28])  # only tf.TensorSpec([None, 28, 28] is allowed
    img_batch_2 = tf.random.uniform(shape=[50, 28, 28])
    preprocessed_images_1 = shrink(img_batch_1)  # Traces the function.
    print(preprocessed_images_1)  # Tracing Tensor("images:0", shape=(None, 28, 28), dtype=float32)
    preprocessed_images_2 = shrink(img_batch_2)  # Reuses the same concrete function.
    print(preprocessed_images_2)

    img_batch_3 = tf.random.uniform(shape=[2, 2, 2])  # it's not a tf.TensorSpec([None, 28, 28] shape!
    try:
        preprocessed_images = shrink(img_batch_3)  # rejects unexpected types or shapes
    except ValueError as ex:
        print(ex)
        '''
        Python inputs incompatible with input_signature:
          inputs: (
            tf.Tensor(
        [[[0.70369995 0.60471594]
          [0.71637666 0.16824412]]
        
         [[0.6009592  0.1905669 ]
          [0.44507182 0.9715203 ]]], shape=(2, 2, 2), dtype=float32))
          input_signature: (
            TensorSpec(shape=(None, 28, 28), dtype=tf.float32, name=None))'''

    # Using Autograph To Capture Control Flow
    print(add_10(tf.constant(5)))  # tf.Tensor(15, shape=(), dtype=int32)
    print(add_10.python_function(5))  # 15
    print(add_10.get_concrete_function(tf.constant(5)).graph.get_operations())
    '''
    [<tf.Operation 'x' type=Placeholder>, 
    <tf.Operation 'add/y' type=Const>, <tf.Operation 'add' type=AddV2>, 
    <tf.Operation 'add_1/y' type=Const>, <tf.Operation 'add_1' type=AddV2>, 
    <tf.Operation 'add_2/y' type=Const>, <tf.Operation 'add_2' type=AddV2>, 
    <tf.Operation 'add_3/y' type=Const>, <tf.Operation 'add_3' type=AddV2>, 
    <tf.Operation 'add_4/y' type=Const>, <tf.Operation 'add_4' type=AddV2>, 
    <tf.Operation 'add_5/y' type=Const>, <tf.Operation 'add_5' type=AddV2>, 
    <tf.Operation 'add_6/y' type=Const>, <tf.Operation 'add_6' type=AddV2>, 
    <tf.Operation 'add_7/y' type=Const>, <tf.Operation 'add_7' type=AddV2>, 
    <tf.Operation 'add_8/y' type=Const>, <tf.Operation 'add_8' type=AddV2>, 
    <tf.Operation 'add_9/y' type=Const>, <tf.Operation 'add_9' type=AddV2>, 
    <tf.Operation 'Identity' type=Identity>]'''

    print(add_10_condition(tf.constant(5)))  # tf.Tensor(15, shape=(), dtype=int32)
    print(add_10_condition.get_concrete_function(tf.constant(5)).graph.get_operations())
    '''
    [<tf.Operation 'x' type=Placeholder>, 
    <tf.Operation 'Const' type=Const>, <tf.Operation 'while/maximum_iterations' type=Const>, 
    <tf.Operation 'while/loop_counter' type=Const>, <tf.Operation 'while' type=While>, 
    <tf.Operation 'while/Identity' type=Identity>, <tf.Operation 'while/Identity_1' type=Identity>, 
    <tf.Operation 'while/Identity_2' type=Identity>, <tf.Operation 'while/Identity_3' type=Identity>, 
    <tf.Operation 'Identity' type=Identity>]'''

    # Using TF Functions with tf.keras (or Not)
    model = MyModel()
    model.compile(loss=my_mse, optimizer="nadam", metrics=[my_mae])
    model.fit(X_train_scaled, y_train, epochs=2,
              validation_data=(X_valid_scaled, y_valid))
    model.evaluate(X_test_scaled, y_test)

    model = MyModel(dynamic=True)  # set it True to stop converting Python to TF function, so it won't train
    model.compile(loss=my_mse, optimizer="nadam", metrics=[my_mae])
    model.fit(X_train_scaled[:64], y_train[:64], epochs=1,
              validation_data=(X_valid_scaled[:64], y_valid[:64]), verbose=0)
    model.evaluate(X_test_scaled[:64], y_test[:64], verbose=0)

    model = MyModel()
    # set run_eagerly=True to stop converting Python to TF function, so it won't train
    model.compile(loss=my_mse, optimizer="nadam", metrics=[my_mae], run_eagerly=True)
    model.fit(X_train_scaled[:64], y_train[:64], epochs=1,
              validation_data=(X_valid_scaled[:64], y_valid[:64]), verbose=0)
    model.evaluate(X_test_scaled[:64], y_test[:64], verbose=0)