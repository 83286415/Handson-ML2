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
def preprocess(line):
    defs = [0.] * n_inputs + [tf.constant([], dtype=tf.float32)]  # 9 elements: eight 0 and a constant tensor
    print(defs)  # [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, <tf.Tensor 'Const:0' shape=(0,) dtype=float32>]
    fields = tf.io.decode_csv(line, record_defaults=defs)
    x = tf.stack(fields[:-1])
    y = tf.stack(fields[-1:])
    return (x - X_mean) / X_std, y


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

    dataset = tf.data.Dataset.range(10).repeat(3)
    print("-----reshuffle------")
    dataset = dataset.shuffle(buffer_size=3, seed=42,  reshuffle_each_iteration=True).batch(7)
    for item in dataset:
        print(item)  # reshuffle_each_iteration=True make data reshuffled (different) in each batch
        ''' tf.Tensor([0 3 4 2 1 5 8], shape=(7,), dtype=int64)
            tf.Tensor([6 9 7 2 3 1 4], shape=(7,), dtype=int64)
            tf.Tensor([6 0 7 9 0 1 2], shape=(7,), dtype=int64)
            tf.Tensor([8 4 5 5 3 8 9], shape=(7,), dtype=int64)
            tf.Tensor([7 6], shape=(2,), dtype=int64)'''

    # Split the California dataset to multiple CSV files

    # load data
    housing = fetch_california_housing()
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        housing.data, housing.target.reshape(-1, 1), random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_full, y_train_full, random_state=42)

    # build a model
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_mean = scaler.mean_
    X_std = scaler.scale_

    train_data = np.c_[X_train, y_train]  # join two matrix right and left
    valid_data = np.c_[X_valid, y_valid]
    test_data = np.c_[X_test, y_test]
    header_cols = housing.feature_names + ["MedianHouseValue"]  # add a column
    header = ",".join(header_cols)

    # write into files
    train_filepaths = save_to_multiple_csv_files(train_data, "train", header, n_parts=20)
    valid_filepaths = save_to_multiple_csv_files(valid_data, "valid", header, n_parts=10)
    test_filepaths = save_to_multiple_csv_files(test_data, "test", header, n_parts=10)

    print(pd.read_csv(train_filepaths[0]).head())
    '''    MedInc  HouseAge  AveRooms  ...  Latitude  Longitude  MedianHouseValue
        0  3.5214      15.0  3.049945  ...     37.63    -122.43             1.442
        1  5.3275       5.0  6.490060  ...     33.69    -117.39             1.687
        2  3.1000      29.0  7.542373  ...     38.44    -122.98             1.621
        3  7.1736      12.0  6.289003  ...     33.55    -117.70             2.621
        4  2.0549      13.0  5.312457  ...     33.93    -116.93             0.956'''
    print(train_filepaths)
    # ['D:\\AI\\handson-ml2-master\\datasets\\housing\\my_train_00.csv',
    # 'D:\\AI\\handson-ml2-master\\datasets\\housing\\my_train_01.csv',
    # 'D:\\AI\\handson-ml2-master\\datasets\\housing\\my_train_02.csv',
    # 'D:\\AI\\handson-ml2-master\\datasets\\housing\\my_train_03.csv',
    # 'D:\\AI\\handson-ml2-master\\datasets\\housing\\my_train_04.csv',
    # 'D:\\AI\\handson-ml2-master\\datasets\\housing\\my_train_05.csv',
    # 'D:\\AI\\handson-ml2-master\\datasets\\housing\\my_train_06.csv',
    # 'D:\\AI\\handson-ml2-master\\datasets\\housing\\my_train_07.csv',
    # 'D:\\AI\\handson-ml2-master\\datasets\\housing\\my_train_08.csv',
    # 'D:\\AI\\handson-ml2-master\\datasets\\housing\\my_train_09.csv',
    # 'D:\\AI\\handson-ml2-master\\datasets\\housing\\my_train_10.csv',
    # 'D:\\AI\\handson-ml2-master\\datasets\\housing\\my_train_11.csv',
    # 'D:\\AI\\handson-ml2-master\\datasets\\housing\\my_train_12.csv',
    # 'D:\\AI\\handson-ml2-master\\datasets\\housing\\my_train_13.csv',
    # 'D:\\AI\\handson-ml2-master\\datasets\\housing\\my_train_14.csv',
    # 'D:\\AI\\handson-ml2-master\\datasets\\housing\\my_train_15.csv',
    # 'D:\\AI\\handson-ml2-master\\datasets\\housing\\my_train_16.csv',
    # 'D:\\AI\\handson-ml2-master\\datasets\\housing\\my_train_17.csv',
    # 'D:\\AI\\handson-ml2-master\\datasets\\housing\\my_train_18.csv',
    # 'D:\\AI\\handson-ml2-master\\datasets\\housing\\my_train_19.csv']

    # Building an Input Pipeline

    filepath_dataset = tf.data.Dataset.list_files(train_filepaths, seed=42)  # make the file paths a data set
    # list_files(shuffle=False) if do NOT want file names shuffled randomly

    n_readers = 5
    dataset = filepath_dataset.interleave(lambda filepath: tf.data.TextLineDataset(filepath).skip(1),
                                          cycle_length=n_readers)
    # interleave: cross the dataset reset by the map function lambda
    # TextLineDataset: create 5 data sets whose info from 5 files in filepath list.
    # cycle_length=AUTOTUNE, controls the number of input elements that are processed concurrently. AUTOTUNE: full CPU
    # skip: ignore the first row in files

    for line in dataset.take(5):
        print(line.numpy())
        ''' b'4.2083,44.0,5.323204419889502,0.9171270718232044,846.0,2.3370165745856353,37.47,-122.2,2.782'
            b'4.1812,52.0,5.701388888888889,0.9965277777777778,692.0,2.4027777777777777,33.73,-118.31,3.215'
            b'3.6875,44.0,4.524475524475524,0.993006993006993,457.0,3.195804195804196,34.04,-118.15,1.625'
            b'3.3456,37.0,4.514084507042254,0.9084507042253521,458.0,3.2253521126760565,36.67,-121.7,2.526'
            b'3.5214,15.0,3.0499445061043287,1.106548279689234,1447.0,1.6059933407325193,37.63,-122.43,1.442'''

    record_defaults = [0, np.nan, tf.constant(np.nan, dtype=tf.float64), "Hello", tf.constant([])]
    parsed_fields = tf.io.decode_csv('1,2,3,4,5', record_defaults)
    print(parsed_fields)
    # [<tf.Tensor: id=234, shape=(), dtype=int32, numpy=1>, <tf.Tensor: id=235, shape=(), dtype=float32, numpy=2.0>,
    # <tf.Tensor: id=236, shape=(), dtype=float64, numpy=3.0>, <tf.Tensor: id=237, shape=(), dtype=string, numpy=b'4'>,
    # <tf.Tensor: id=238, shape=(), dtype=float32, numpy=5.0>]

    parsed_fields = tf.io.decode_csv(',,,,5', record_defaults)
    print(parsed_fields)
    # [<tf.Tensor: id=243, shape=(), dtype=int32, numpy=0>, <tf.Tensor: id=244, shape=(), dtype=float32, numpy=nan>,
    # <tf.Tensor: id=245, shape=(), dtype=float64, numpy=nan>, <tf.Tensor: id=246, shape=(), dtype=string,
    # numpy=b'Hello'>, <tf.Tensor: id=247, shape=(), dtype=float32, numpy=5.0>]

    try:
        parsed_fields = tf.io.decode_csv(',,,,', record_defaults)
    except tf.errors.InvalidArgumentError as ex:
        print(ex)  # Field 4 is required but missing in record 0! [Op:DecodeCSV]

    try:
        parsed_fields = tf.io.decode_csv('1,2,3,4,5,6,7', record_defaults)
    except tf.errors.InvalidArgumentError as ex:
        print(ex)  # Expect 5 fields but have 7 in record 0 [Op:DecodeCSV]

    n_inputs = 8  # X_train.shape[-1]
    print(preprocess(b'4.2083,44.0,5.3232,0.9171,846.0,2.3370,37.47,-122.2,2.782'))
    # (<tf.Tensor: id=286, shape=(8,), dtype=float32,  numpy=array([ 0.16579157,  1.216324  , -0.05204565, -0.39215982,
    # -0.5277444 , -0.2633488 ,  0.8543046 , -1.3072058 ], dtype=float32)>, <tf.Tensor: id=287, shape=(1,),
    # dtype=float32, numpy=array([2.782], dtype=float32)>)

