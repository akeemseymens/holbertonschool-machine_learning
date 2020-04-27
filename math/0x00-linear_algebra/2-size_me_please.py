#!/usr/bin/env python3
"""Calculates the shape of a matrix."""


def matrix_shape(matrix):
    """Use numpy function."""
    if not matrix:
        return None
    if type(matrix[0]) is not list:
        return [len(matrix)]
    return [len(matrix)] + matrix_shape(matrix[0])
