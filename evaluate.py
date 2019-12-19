"""Operations related to evaluating predictions.
"""

import numpy as np


def mse(y, y_pred):
    return np.mean((y - y_pred) ** 2)


def rmse(y, y_pred):
    return np.sqrt(mse(y, y_pred))


def mae(y, y_pred):
    return np.mean(np.fabs(y - y_pred))

def hr(y, y_pred, error = 0.05):
    dif = np.fabs(y - y_pred)
    dif[dif > error] = 0
    dif[dif > 0] = 1
    return np.mean(dif)