"""Classes and operations related to processing data.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf

class BatchGenerator(object):
    def __init__(self, x, y=None, batch_size=1024, shuffle=True):
        assert(y is None or x.shape[0] == y.shape[0])
        self.x = x
        self.y = y
        self.length = x.shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle


    def next(self):
        start = end = 0
        length = self.length
        batch_size = self.batch_size

        if self.shuffle:
            permutation = np.random.permutation(length)
            self.x = self.x[permutation]
            self.y = self.y[permutation]

        flag = False
        while not flag:
            end += batch_size
            if end > length:
                end = length - 1
                flag = True
            yield self._get_batch(start, end)
            start = end


    def _get_batch(self, start, end):
        if self.y is not None:
            return self.x[start:end], self.y[start:end]
        else:
            return self.x[start:end]


def save_model(sess, model_dir):
    saver = tf.compat.v1.train.Saver()
    saver.save(sess, model_dir)


def load_model(sess, model_path):
    tensor_names = ['placeholder/users:0', 'placeholder/items:0',
                    'placeholder/ratings:0', 'prediction/pred:0']
    operation_names = ['optimizer/optimizer']

    saver = tf.train.import_meta_graph(model_path + '.meta')
    saver.restore(sess, model_path)

    for name in tensor_names:
        attr = '_' + name.split('/')[1].split(':')[0]
        setattr(self, attr, tf.get_default_graph().get_tensor_by_name(name))

    for name in operation_names:
        attr = '_' + name.split('/')[1].split(':')[0]
        setattr(self, attr, tf.get_default_graph(
        ).get_operation_by_name(name))


def part_data(data_x, data_y, partitions):
    assert(data_y is None or data_x.shape[0] == data_y.shape[0])
    partitioned_data = []
    data_len = data_x.shape[0]
    permutation = np.random.permutation(data_len)
    data_x = data_x[permutation]
    data_y = data_y[permutation]
    start = 0
    for part in partitions:
        end = int(start + part * data_len)
        partitioned_data.append(data_x[start:end])
        partitioned_data.append(data_y[start:end])
        start = end
    return tuple(partitioned_data)


def load_ml1m():
    df = pd.read_csv("data/ml-1m.dat", sep='::', header=None, engine='python')
    x = df.iloc[:, :2].values
    y = df.iloc[:, 2].values
    y = (y - np.mean(y))/5
    return x, y


def load_ml100k():
    df = pd.read_csv("data/ml-100k.dat", sep='\t', header=None)
    x = df.iloc[:, :2].values
    y = df.iloc[:, 2].values
    y = (y - np.mean(y))/5
    return x, y


def generate_data(num_user = 6040,num_item = 400):
    u = np.random.random((num_user, 1))
    v = np.random.random((num_item, 1))
    mat = np.dot(u, np.transpose(v))
    #转换格式
    l = [[], []]
    for u in range(num_user):
        l[0].extend([u for i in range(num_item)])
        l[1].extend(list(range(num_item)))
    x = np.array(l).T
    y = np.reshape(mat, num_user*num_item)
    #打乱
    data_len = x.shape[0]
    permutation = np.random.permutation(data_len)
    x = x[permutation]
    y = y[permutation]
    return x, y