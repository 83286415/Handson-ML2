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

    # from Numpy
    np_array = np.array([2., 4., 5.])
    from_np = tf.constant(np_array)
    print(from_np.numpy)  # np array -> tensor
    # <bound method _EagerTensorBase.numpy of <tf.Tensor: id=35, shape=(3,), dtype=float64, numpy=array([2., 4., 5.])>>
    # alert: notice its dtype is float64 in np. So make sure set dtype=tf.float32 when change it into a tensor.

    # to Numpy
    to_np = from_np.numpy()  # two ways to transfer tensor -> np array
    print(to_np)  # array([2. 4. 5.], dtype=float64)
    to_np = np.array(from_np, dtype='float32')
    print(to_np.dtype)  # dtype=float32

    # Conflicting Types

    try:
        tf.constant(2.0) + tf.constant(40)  # float + int32 = error
    except tf.errors.InvalidArgumentError as ex:  # different types make error when their tensor add
        print(ex)
        # cannot compute AddV2 as input #1(zero-based) was expected to be a float tensor
        # but is a int32 tensor [Op:AddV2] name: add/

    try:
        tf.constant(2.0) + tf.constant(40., dtype=tf.float64)  # float + double float = error!
    except tf.errors.InvalidArgumentError as ex:  # different precisions make error when their tensor add
        print(ex)
        # cannot compute AddV2 as input #1(zero-based) was expected to be a float tensor
        # but is a double tensor [Op:AddV2] name: add/

    t2 = tf.constant(40., dtype=tf.float64)  # change the default precision 32bit to 64bit
    t_add_result = tf.constant(2.0) + tf.cast(t2, tf.float32)  # change back to 32bit and add
    print(t_add_result)  # tf.Tensor(42.0, shape=(), dtype=float32) <dtype: 'float32'>

    # Variables

    v = tf.Variable([[1., 2., 3.], [4., 5., 6.]])

    print(v.assign(2 * v))
    # <tf.Variable 'UnreadVariable' shape=(2, 3) dtype=float32,
    # numpy=array([[ 2.,  4.,  6.], [ 8., 10., 12.]], dtype=float32)>  default dtype is float32 not float64!

    print(v[0, 1].assign(42))
    # <tf.Variable 'UnreadVariable' shape=(2, 3) dtype=float32,
    # numpy=array([[ 2., 42.,  6.], [ 8., 10., 12.]], dtype=float32)>

    print(v[:, 2].assign([0., 1.]))
    # <tf.Variable 'UnreadVariable' shape=(2, 3) dtype=float32,
    # numpy=array([[ 2., 42.,  0.], [ 8., 10.,  1.]], dtype=float32)>

    try:
        v[1] = [7., 8., 9.]  # cannot modify values without assign(), scatter_update() or scatter_nd_update() method
    except TypeError as ex:
        print(ex)  # 'ResourceVariable' object does not support item assignment

    print(v.scatter_nd_update(indices=[[0, 0], [1, 2]], updates=[100., 200.]))
    # <tf.Variable 'UnreadVariable' shape=(2, 3) dtype=float32,
    # numpy=array([[100.,  42.,   0.], [  8.,  10., 200.]], dtype=float32)>

    sparse_delta = tf.IndexedSlices(values=[[1., 2., 3.], [4., 5., 6.]], indices=[1, 0])  # prepare index and values
    print(v.scatter_update(sparse_delta))
    # <tf.Variable 'UnreadVariable' shape=(2, 3) dtype=float32,
    # numpy=array([[4., 5., 6.], [1., 2., 3.]], dtype=float32)>

    # Other Data Structures: refer to Appendix F in the book

    # Strings

    print(tf.constant(b"hello world"))  # tf.Tensor(b'hello world', shape=(), dtype=string)  No shape for string tensor
    print(tf.constant("café"))  # tf.Tensor(b'caf\xc3\xa9', shape=(), dtype=string)  unicode -> utf8
    print(tf.constant("汉字"))  # tf.Tensor(b'\xe6\xb1\x89\xe5\xad\x97', shape=(), dtype=string)  unicode -> utf8

    # string no shape, but int32 gets shape
    char_unicode_2_int32 = tf.constant([ord(c) for c in "café"])  # char(unicode) -> int32
    print(char_unicode_2_int32)  # tf.Tensor([ 99  97 102 233], shape=(4,), dtype=int32)  Got shape for int32

    # switch int32/unicode(utf8) by tf.strings
    int32_2_utf8 = tf.strings.unicode_encode(char_unicode_2_int32, "UTF-8")  # int32 -> unicode but into utf8 auto
    print(int32_2_utf8)  # tf.Tensor(b'caf\xc3\xa9', shape=(), dtype=string)  # utf8
    print(tf.strings.length(int32_2_utf8, unit="UTF8_CHAR"))  # get length: tf.Tensor(4, shape=(), dtype=int32)

    unicode_2_int32 = tf.strings.unicode_decode(int32_2_utf8, "UTF-8")  # utf8(unicode) decoded into int32
    print(unicode_2_int32)
    # <tf.Tensor: id=50, shape=(4,), dtype=int32, numpy=array([ 99,  97, 102, 233], dtype=int32)>

    # String Arrays

    # create a string array
    string_array = tf.constant(["Café", "Coffee", "caffè", "咖啡"])  # an array of strings above
    print(string_array)  # string array get shape of the array not strings
    # tf.Tensor([b'Caf\xc3\xa9' b'Coffee' b'caff\xc3\xa8' b'\xe5\x92\x96\xe5\x95\xa1'], shape=(4,), dtype=string)
    print(tf.strings.length(string_array, unit="UTF8_CHAR"))  # length: tf.Tensor([4 6 5 2], shape=(4,), dtype=int32)

    # transfer string array (tensor) into a Ragged Tensor with int32 type by tf.strings
    string_array_2_int32_ragged_tensor = tf.strings.unicode_decode(string_array, "UTF8")
    print(string_array_2_int32_ragged_tensor)
    # <tf.RaggedTensor [[67, 97, 102, 233], [67, 111, 102, 102, 101, 101], [99, 97, 102, 102, 232], [21654, 21857]]>

    # Ragged Tensors

    # difference between tensor and ragged tensor
    print(string_array_2_int32_ragged_tensor[1])  # tf.Tensor([ 67 111 102 102 101 101], shape=(6,), dtype=int32)
    print(string_array_2_int32_ragged_tensor[1:3])
    # <tf.RaggedTensor [[67, 111, 102, 102, 101, 101], [99, 97, 102, 102, 232]]>

    # create ragged tensor and its operation
    ragged_tensor = tf.ragged.constant([[65, 66], [], [67]])
    print(tf.concat([string_array_2_int32_ragged_tensor, ragged_tensor], axis=0))
    # <tf.RaggedTensor
    # [[67, 97, 102, 233], [67, 111, 102, 102, 101, 101], [99, 97, 102, 102, 232], [21654, 21857], [65, 66], [], [67]]>

    ragged_tensor_2 = tf.ragged.constant([[68, 69, 70], [71], [72, 73]])
    print(tf.concat([ragged_tensor, ragged_tensor_2], axis=1))  # the last axis: concat each tensor by order
    # <tf.RaggedTensor [[65, 66, 68, 69, 70], [71], [67, 72, 73]]>

    # ragged tensor to tensor in unicode
    ragged_tensor_2_string_array = tf.strings.unicode_encode(ragged_tensor_2, "UTF-8")
    print(ragged_tensor_2_string_array)  # tf.Tensor([b'DEF' b'G' b'HI'], shape=(3,), dtype=string)

    # ragged tensor to tensor in int32
    ragged_tensor_2_int32_tensor = string_array_2_int32_ragged_tensor.to_tensor()
    print(ragged_tensor_2_int32_tensor)  # zeros padding the ragged tensors for their different shapes
    '''tf.Tensor(
                [[   67    97   102   233     0     0]
                 [   67   111   102   102   101   101]
                 [   99    97   102   102   232     0]
                 [21654 21857     0     0     0     0]], shape=(4, 6), dtype=int32)'''

    # Sparse Tensor

    # define a sparse tensor
    sparse_tensor = tf.SparseTensor(indices=[[0, 1], [1, 0], [2, 3]], values=[1., 2., 3.], dense_shape=[3, 4])
    print(sparse_tensor)
    # SparseTensor(
    # indices=tf.Tensor([[0 1] [1 0] [2 3]], shape=(3, 2), dtype=int64),
    # values=tf.Tensor([1. 2. 3.], shape=(3,), dtype=float32),
    # dense_shape=tf.Tensor([3 4], shape=(2,), dtype=int64))

    # sparse to dense
    sparse_2_dense = tf.sparse.to_dense(sparse_tensor)
    print(sparse_2_dense)
    '''tf.Tensor(
                [[0. 1. 0. 0.]
                 [2. 0. 0. 0.]
                 [0. 0. 0. 3.]], shape=(3, 4), dtype=float32)'''  # dense can be shown but sparse cannot. they are same.

    # operations
    sparse_tensor_2 = sparse_tensor * 2  # sparse tensor: the type of tensors can operated
    print(sparse_tensor_2)  # values are doubled
    # SparseTensor(
    # indices=tf.Tensor([[0 1] [1 0] [2 3]], shape=(3, 2), dtype=int64),
    # values=tf.Tensor([2. 4. 6.], shape=(3,), dtype=float32),
    # dense_shape=tf.Tensor([3 4], shape=(2,), dtype=int64))

    try:
        sparse_2_dense_3 = sparse_tensor_2 + 1.  # different types cannot operated
    except TypeError as ex:
        print(ex)  # unsupported operand type(s) for +: 'SparseTensor' and 'float'

    tensor_set = tf.constant([[10., 20.], [30., 40.], [50., 60.], [70., 80.]])
    sparse_matmul = tf.sparse.sparse_dense_matmul(sparse_tensor, tensor_set)
    print(sparse_matmul)  # shape (3, 4) * shape (4, 2) = shape (3, 2)
    '''tf.Tensor(
                [[ 30.  40.]
                 [ 20.  40.]
                 [210. 240.]], shape=(3, 2), dtype=float32)'''

    # define another sparse tensor
    another_sparse_tensor = tf.SparseTensor(indices=[[0, 2], [0, 1]], values=[1., 2.], dense_shape=[3, 4])
    print(another_sparse_tensor)  # note: there is an indices order error will be caught below.
    # SparseTensor(
    # indices=tf.Tensor([[0 2][0 1]], shape=(2, 2), dtype=int64),
    # values=tf.Tensor([1. 2.], shape=(2,), dtype=float32),
    # dense_shape=tf.Tensor([3 4], shape=(2,), dtype=int64))

    try:
        another_sparse_2_dense = tf.sparse.to_dense(another_sparse_tensor)
        print(another_sparse_2_dense)
    except tf.errors.InvalidArgumentError as ex:
        print(ex)  # indices[1] = [0,1] is out of order [Op:SparseToDense]
        # note: the location of the second index [0, 1] has been already 0, so cannot set it 2. then error occurs.
        # we can change the sparse indices like this [[0, 2], [1, 1]] to avoid this error or reorder it as below:

    another_sparse_reorder = tf.sparse.reorder(another_sparse_tensor)
    # reorder: does NOT change the indices params but make latter elements over-write zeros located in tensor
    another_sparse_2_dense = tf.sparse.to_dense(another_sparse_reorder)
    print(another_sparse_2_dense)
    '''tf.Tensor(
                [[0. 2. 1. 0.]
                 [0. 0. 0. 0.]
                 [0. 0. 0. 0.]], shape=(3, 4), dtype=float32)'''

    # Sets

    set1 = tf.constant([[2, 3, 5, 7], [7, 9, 0, 0]])
    set2 = tf.constant([[4, 5, 6], [9, 10, 0]])
    print(set2)  # set is a tensor with more than one sub-tensors
    '''tf.Tensor(
                [[ 4  5  6]
                 [ 9 10  0]], shape=(2, 3), dtype=int32)'''

    # sets union
    set_1_union_2 = tf.sets.union(set1, set2)
    print(set_1_union_2)  # sets union is a sparse tensor
    '''
    SparseTensor(
        indices=tf.Tensor(
                        [[0 0]
                         [0 1]
                         [0 2]
                         [0 3]
                         [0 4]
                         [0 5]
                         [1 0]
                         [1 1]
                         [1 2]
                         [1 3]], shape=(10, 2), dtype=int64), 
        values=tf.Tensor([ 2  3  4  5  6  7  0  7  9 10], shape=(10,), dtype=int32), 
        dense_shape=tf.Tensor([2 6], shape=(2,), dtype=int64))'''

    set_union_2_dense = tf.sparse.to_dense(set_1_union_2)
    print(set_union_2_dense)
    '''tf.Tensor(
                [[ 2  3  4  5  6  7]
                 [ 0  7  9 10  0  0]], shape=(2, 6), dtype=int32)'''

    # sets difference
    set_1_difference_2 = tf.sets.difference(set1, set2)
    print(set_1_difference_2)  # sparse tensor
    '''SparseTensor(
            indices=tf.Tensor(
                            [[0 0]
                             [0 1]
                             [0 2]
                             [1 0]], shape=(4, 2), dtype=int64), 
            values=tf.Tensor([2 3 7 7], shape=(4,), dtype=int32), 
            dense_shape=tf.Tensor([2 3], shape=(2,), dtype=int64))'''

    set_difference_2_dense = tf.sparse.to_dense(set_1_difference_2)
    print(set_difference_2_dense)
    '''tf.Tensor(
                [[2 3 7]
                 [7 0 0]], shape=(2, 3), dtype=int32)'''

    # sets intersection
    set_1_intersection_2 = tf.sets.intersection(set1, set2)
    print(set_1_intersection_2)
    '''SparseTensor(
                indices=tf.Tensor(
                                [[0 0]
                                 [1 0]
                                 [1 1]], shape=(3, 2), dtype=int64), 
                values=tf.Tensor([5 0 9], shape=(3,), dtype=int32), 
                dense_shape=tf.Tensor([2 2], shape=(2,), dtype=int64))'''

    set_intersection_2_dense = tf.sparse.to_dense(set_1_intersection_2)
    print(set_intersection_2_dense)
    '''tf.Tensor(
                [[5 0]
                 [0 9]], shape=(2, 2), dtype=int32)'''

    # Tensor Array

    # define a tensor array
    array = tf.TensorArray(dtype=tf.float32, size=3, clear_after_read=False)
    array = array.write(0, tf.constant([1., 2.]))
    array = array.write(1, tf.constant([3., 10.]))
    array = array.write(2, tf.constant([5., 7.]))

    print(array.read(1))  # tf.Tensor([ 3. 10.], shape=(2,), dtype=float32)
    print(array.stack())  # return a stacked tensor
    '''tf.Tensor(
                [[1. 2.]
                 [ 3. 10.] # if the para clear_after_read=True in TensorArray(), this value should be [0. 0.]
                 [5. 7.]], shape=(3, 2), dtype=float32)'''

    mean, variance = tf.nn.moments(array.stack(), axes=0)  # return mean and variance
    # axes=1 return mean and variance of each sub-tensor in array.stack()
    print(mean)  # tf.Tensor([3.        6.3333335], shape=(2,), dtype=float32)
    print(variance)  # tf.Tensor([ 2.6666667 10.888888 ], shape=(2,), dtype=float32)
