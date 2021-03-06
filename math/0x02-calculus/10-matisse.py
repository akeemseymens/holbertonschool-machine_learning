#!/usr/bin/env python3
"""Calculate the derivative of a polynomial."""


def poly_derivative(poly):
    """Calculate the derivative of a polynomial."""
    if poly and isinstance(poly, list) and all(
            isinstance(x, (int, float)) for x in poly):
        result = [poly[i] * i for i in range(1, len(poly))]
        if not len(result):
            return [0]
        return result
    return None
