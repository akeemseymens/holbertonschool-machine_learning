#!/usr/bin/env python3
"""Converts a numeric label vector into a one-hot matrix."""

import numpy as np


def one_hot_encode(Y, classes):
    """
    Y is numpy.ndarray shape (m,) contains numeric labels.
    m is number of examples.
    classes is max number classes found in Y.
    """
    if not isinstance(Y, np.ndarray) or len(Y) == 0:
        return None
    if not isinstance(classes, int) or classes < np.max(Y) + 1:
        return None
    arr = np.zeros((classes, Y.shape[0]))
    for cl, m in enumerate(Y):
        arr[m][cl] = 1
    return arr
