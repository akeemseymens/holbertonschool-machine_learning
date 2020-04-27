#!/usr/bin/env python3
"""Transposes a matrix."""


def matrix_transpose(matrix):
    """Function that transposes a matrix."""
    zipped_rows = zip(*matrix)
    transpose_matrix = [list(row) for row in zipped_rows]
    return transpose_matrix
