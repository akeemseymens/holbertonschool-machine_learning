#!/usr/bin/env python3
"""Add two matrices together."""


def add_matrices2D(mat1, mat2):
    """Add two matrices together."""
    if len(mat1) == len(mat2) and len(mat1[0]) == len(mat2[0]):
        return [[mat1[i][j] + mat2[i][j] for j in range(len(mat1[0]))]
                for i in range(len(mat1))]
    else:
        return None
