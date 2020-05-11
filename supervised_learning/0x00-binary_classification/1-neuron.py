#!/usr/bin/env python3
"""Class for Neuron"""
import numpy as np


class Neuron:
    """Defines a single neuron performing binary classifications."""

    def __init__(self, nx):
        """Intialize the class."""
        if type(nx) is not int:
            raise TypeError("nx must be a integer")
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0
    
    @property
    def W(self):
        """Returns weights"""
        return self.__W

    @property
    def b(self):
        """Returns bias"""
        return self.__b

    @property
    def A(self):
        """Returns activation output"""
        return self.__A
