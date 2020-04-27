#!/usr/bin/env python3
"""Multiplying two matrices together."""


def mat_mul(mat1, mat2):
    """Multiply two matrices using list comprehension."""
    new_matrix = []
    if len(mat1[0]) == len(mat2):
        new_matrix = [[sum(i * j for i, j in zip(row, col))
                      for col in zip(*mat2)] for row in mat1]
        return (new_matrix)
    else:
        return None
