import os
import numpy as np
import tensorflow as tf
try:
    from tensorflow.keras import utils
except:
    from tensorflow.contrib.keras import utils
from utils import *
from evaluate import *


class SVD():
    """Collaborative filtering model based on SVD algorithm.
    """
    def __init__(self, sess, x, y, epochs=20, batch_size=1024, num_factors = 15):
        assert(x.shape[0] == y.shape[0])
        assert(x.shape[1] == 2)
        self.num_users = np.max(x[:, 0]) + 1
        self.num_items = np.max(x[:, 1]) + 1
        self.min_value = np.min(y)
        self.max_value = np.max(y)
        self.num_factors = num_factors
        self.epochs = epochs
        self.batch_size = batch_size

        self._sess = sess
        self.reg_b_u = 0.0001
        self.reg_b_i = 0.0001
        self.reg_p_u = 0.005
        self.reg_q_i = 0.005

        self.x_train, self.y_train, self.x_valid, self.y_valid,\
            self.x_test, self.y_test = part_data(x, y, [0.6, 0.2, 0.2])

        #构建迭代结构
        _mu = tf.constant(np.mean(self.y_train), shape=[], dtype=tf.float32)
        self._users = tf.compat.v1.placeholder(tf.int32, shape=[None, ], name='users')
        self._items = tf.compat.v1.placeholder(tf.int32, shape=[None, ], name='items')
        self._ratings = tf.compat.v1.placeholder(tf.float32, shape=[None, ], name='ratings')

        #user embeddings
        user_embeddings = tf.compat.v1.get_variable(name='user_embedding', shape=[self.num_users, self.num_factors],
            initializer=tf.contrib.layers.xavier_initializer(),
            regularizer=tf.contrib.layers.l2_regularizer(self.reg_p_u))

        user_bias = tf.compat.v1.get_variable(name='user_bias', shape=[self.num_users, ],
            initializer=tf.contrib.layers.xavier_initializer(),
            regularizer=tf.contrib.layers.l2_regularizer(self.reg_b_u))

        p_u = tf.nn.embedding_lookup(user_embeddings, self._users, name='p_u')
        b_u = tf.nn.embedding_lookup(user_bias, self._users, name='b_u')
        
        #item embeddings
        item_embeddings = tf.compat.v1.get_variable(name='item_embedding',shape=[self.num_items, self.num_factors],
            initializer=tf.contrib.layers.xavier_initializer(),
            regularizer=tf.contrib.layers.l2_regularizer(self.reg_q_i))

        item_bias = tf.compat.v1.get_variable(name='item_bias', shape=[self.num_items,], initializer=tf.contrib.layers.xavier_initializer(),
            regularizer=tf.contrib.layers.l2_regularizer(self.reg_b_i))

        q_i = tf.nn.embedding_lookup(item_embeddings, self._items, name='q_i')
        b_i = tf.nn.embedding_lookup(item_bias, self._items, name='b_i')

        self._pred = tf.reduce_sum(tf.multiply(p_u, q_i), axis=1)
        self._pred = tf.add_n([b_u, b_i, self._pred])
        self._pred = tf.add(self._pred, _mu, name='pred')

        loss = tf.nn.l2_loss(tf.subtract(self._ratings, self._pred), name='loss')
        objective = tf.add(loss, tf.add_n(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES)))
        try:
            self._optimizer = tf.contrib.keras.optimizers.Nadam().minimize(objective, name='optimizer')
        except:
            self._optimizer = tf.compat.v1.train.AdamOptimizer().minimize(objective, name='optimizer')




    def train(self):
        train_gen = BatchGenerator(self.x_train, self.y_train, self.batch_size)
        steps_per_epoch = np.ceil(train_gen.length / self.batch_size).astype(int)

        self._sess.run(tf.compat.v1.global_variables_initializer())

        for e in range(1, self.epochs + 1):
            print('Epoch {}/{}'.format(e, self.epochs))

            pbar = utils.Progbar(steps_per_epoch)

            for step, batch in enumerate(train_gen.next(), 1):
                users = batch[0][:, 0]
                items = batch[0][:, 1]
                ratings = batch[1]

                self._sess.run(
                    self._optimizer,
                    feed_dict={
                        self._users: users,
                        self._items: items,
                        self._ratings: ratings
                    })

                pred = self.predict(batch[0])
                update_values = [
                    ('rmse', rmse(ratings, pred)),
                    ('mae', mae(ratings, pred)),
                    ('hr', hr(ratings, pred))
                ]

                if self.x_valid is not None and step == steps_per_epoch:
                    valid_pred = self.predict(self.x_valid)
                    update_values += [
                        ('val_rmse', rmse(self.y_valid, valid_pred)),
                        ('val_mae', mae(self.y_valid, valid_pred)),
                        ('val_hr', hr(self.y_valid, valid_pred))
                    ]

                pbar.update(step, values=update_values)

        y_pred = self.predict(self.x_test)
        print('rmse: {}, mae: {}, hr:{}'.format(rmse(self.y_test, y_pred), mae(self.y_test, y_pred), hr(self.y_test, y_pred)))


    def predict(self, x):
        users, items = x[:,0], x[:, 1]
        pred = self._sess.run(self._pred,feed_dict={
            self._users: users, self._items: items})
        pred = pred.clip(min=self.min_value, max=self.max_value)
        return pred
