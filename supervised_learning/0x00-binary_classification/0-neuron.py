#!/usr/bin/env python3
"""Class for Neuron."""
import numpy as np


class Neuron:
    """Defines a single neuron performing binary classifications."""

    def __init__(self, nx):
        """Intialize the class."""
        if type(nx) is not int:
            raise TypeError("nx must be a integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.W = np.random.normal(size=(1, nx))
        self.b = 0
        self.A = 0
