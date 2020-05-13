#!/usr/bin/env python3
"""function converts one-hot matrix to vector of labels"""

import numpy as np


def one_hot_decode(one_hot):
    """
    one_hot is encoded numpy.ndarray w/ shape
    (classes, m)
    """

    if not isinstance(one_hot, np.ndarray) or len(one_hot.shape) != 2:
        return None
    if not np.where((one_hot == 0) | (one_hot == 1), True, False).all():
        return None
    if np.sum(one_hot) != one_hot.shape[1]:
        return None
    return np.argmax(one_hot, axis=0)
    