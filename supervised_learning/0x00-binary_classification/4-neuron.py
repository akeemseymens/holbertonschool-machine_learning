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
        """Returns active"""
        return self.__A

    def forward_prop(self, X):
        """forwared pass"""
        M = np.matmul(self.__W, X) + self.__b
        self.__A = self.sigmoid(M)
        return self.__A

    def sigmoid(self, X):
        """sigmoid function"""
        return 1.0/(1.0 + np.exp(-X))

    def cost(self, Y, A):
        """Cost of Model using logistic regression"""
        return -(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)).mean()

    def evaluate(self, X, Y):
        """evaluates neuron predictions"""
        self.forward_prop(X)
        return np.round(self.__A).astype(int), self.cost(Y, self.__A)
