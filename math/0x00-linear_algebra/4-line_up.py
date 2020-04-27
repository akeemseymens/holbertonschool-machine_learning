#!/usr/bin/env python3
"""Adding two arrays together."""


def add_arrays(arr1, arr2):
    """Adding two arrays together."""
    if len(arr1) == len(arr2):
        return [arr1[i] + arr2[i] for i in range(len(arr1))]
    else:
        return None
