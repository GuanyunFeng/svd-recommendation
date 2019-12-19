import numpy as np
import tensorflow as tf
import os
from evaluate import *
from utils import *
from svd import SVD
from svdpp import SVDPP



if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # x[i, :] contains the user-item pair, and y[i] is the corresponding rating.
    x, y = load_ml1m()
    #x, y = generate_data()

    with tf.compat.v1.Session() as sess:
        #model = SVDPP(sess, x, y, num_factors = 15, dual=False)
        model = SVDPP(sess, x, y, num_factors = 15, dual=True)
        #model = SVD(sess, x, y, epochs=20, batch_size=1024, num_factors = 15)
        model.train()
        # Save model
        save_model(sess, 'model/svd')