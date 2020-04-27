#!/usr/bin/env python3
"""Getting Cozy illustrates how to concatenate over a specif axis."""


def cat_matrices2D(mat1, mat2, axis=0):
    """Adding two matrices on an a specific axis."""
    if axis == 1:
        if len(mat1) != len(mat2):
            return None
        return [i + j for i, j in zip(mat1, mat2)]
    else:
        if len(mat1[0]) != len(mat2[0]):
            return None
        return [i.copy() for i in mat1] + [j.copy() for j in mat2]
