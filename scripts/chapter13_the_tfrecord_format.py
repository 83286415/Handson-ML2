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

# chapter import
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

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
TFRECORD_PATH = os.path.join(PROJECT_ROOT_DIR, "TFrecord", CHAPTER_ID)
os.makedirs(TFRECORD_PATH, exist_ok=True)

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


def save_to_multiple_csv_files(data, name_prefix, header=None, n_parts=10):
    housing_dir = os.path.join(PROJECT_ROOT_DIR, "datasets", "housing")
    os.makedirs(housing_dir, exist_ok=True)
    path_format = os.path.join(housing_dir, "my_{}_{:02d}.csv")

    filepaths = []
    m = len(data)
    for file_idx, row_indices in enumerate(np.array_split(np.arange(m), n_parts)):
        part_csv = path_format.format(name_prefix, file_idx)
        filepaths.append(part_csv)
        with open(part_csv, "wt", encoding="utf-8") as f:
            if header is not None:
                f.write(header)
                f.write("\n")
            for row_idx in row_indices:
                f.write(",".join([repr(col) for col in data[row_idx]]))
                f.write("\n")
    return filepaths


@tf.function
def preprocess(line):  # line: a string from csv file, each element is separated by a comma
    defs = [0.] * n_inputs + [tf.constant([], dtype=tf.float32)]  # 9 elements: eight 0 and a constant tensor
    print(defs)  # [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, <tf.Tensor 'Const:0' shape=(0,) dtype=float32>]
    fields = tf.io.decode_csv(line, record_defaults=defs)  # return a list of scalar tensor
    x = tf.stack(fields[:-1])  # x should be a 1D tensor but the last tensor in fields as it is the target y
    y = tf.stack(fields[-1:])
    return (x - X_mean) / X_std, y


def csv_reader_dataset(filepaths, repeat=1, n_readers=5,
                       n_read_threads=None, shuffle_buffer_size=10000,
                       n_parse_threads=5, batch_size=32):  # shuffle + preprocess + repeat + batch + prefetch
    dataset = tf.data.Dataset.list_files(filepaths).repeat(repeat)
    dataset = dataset.interleave(
        lambda filepath: tf.data.TextLineDataset(filepath).skip(1),
        cycle_length=n_readers, num_parallel_calls=n_read_threads)
    dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.map(preprocess, num_parallel_calls=n_parse_threads)
    dataset = dataset.batch(batch_size)
    return dataset.prefetch(1)


@tf.function
def train(model, n_epochs, batch_size=32,
          n_readers=5, n_read_threads=5, shuffle_buffer_size=10000, n_parse_threads=5):
    train_set = csv_reader_dataset(train_filepaths, repeat=n_epochs, n_readers=n_readers,
                       n_read_threads=n_read_threads, shuffle_buffer_size=shuffle_buffer_size,
                       n_parse_threads=n_parse_threads, batch_size=batch_size)
    optimizer = keras.optimizers.Nadam(lr=0.01)
    loss_fn = keras.losses.mean_squared_error
    n_steps_per_epoch = len(X_train) // batch_size
    total_steps = n_epochs * n_steps_per_epoch
    global_step = 0
    for X_batch, y_batch in train_set.take(total_steps):
        global_step += 1
        if tf.equal(global_step % 100, 0):
            tf.print("\rGlobal step", global_step, "/", total_steps)
        with tf.GradientTape() as tape:
            y_pred = model(X_batch)
            main_loss = tf.reduce_mean(loss_fn(y_batch, y_pred))
            loss = tf.add_n([main_loss] + model.losses)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))


if __name__ == '__main__':

    # some codes in last py: the data api

    # load data
    housing = fetch_california_housing()
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        housing.data, housing.target.reshape(-1, 1), random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_full, y_train_full, random_state=42)

    # config
    n_inputs = 8
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_mean = scaler.mean_
    X_std = scaler.scale_

    # write into files
    train_data = np.c_[X_train, y_train]  # join two matrix right and left
    valid_data = np.c_[X_valid, y_valid]
    test_data = np.c_[X_test, y_test]
    header_cols = housing.feature_names + ["MedianHouseValue"]  # add a column
    header = ",".join(header_cols)

    train_filepaths = save_to_multiple_csv_files(train_data, "train", header, n_parts=20)
    valid_filepaths = save_to_multiple_csv_files(valid_data, "valid", header, n_parts=10)
    test_filepaths = save_to_multiple_csv_files(test_data, "test", header, n_parts=10)

    # The TFRecord binary format

    # write a tfrecord
    tfrecord_file_path = os.path.join(TFRECORD_PATH, "my_data.tfrecord")
    with tf.io.TFRecordWriter(tfrecord_file_path) as f:
        f.write(b"This is the first record")
        f.write(b"And this is the second record")

    # load a tfrecord to a dataset
    dataset = tf.data.TFRecordDataset(tfrecord_file_path, num_parallel_reads=1)  # parallel read
    for item in dataset:
        print(item)  # two tensors
        # tf.Tensor(b'This is the first record', shape=(), dtype=string)
        # tf.Tensor(b'And this is the second record', shape=(), dtype=string)

    # Compressed TFRecord Files

    # write a TFrecord compressed file
    options = tf.io.TFRecordOptions(compression_type="GZIP")  # ZLIB, GZIP, None
    with tf.io.TFRecordWriter("my_compressed.tfrecord", options) as f:
        f.write(b"This is the first record")
        f.write(b"And this is the second record")

    # load a compressed TFrecord file into a dataset
    dataset = tf.data.TFRecordDataset(["my_compressed.tfrecord"], compression_type="GZIP")
