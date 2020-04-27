#!/usr/bin/env python3
"""Concatenates two matrices along specific axis."""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """Return a new array with concatenated two matrices."""
    return np.concatenate((mat1, mat2), axis=axis)
