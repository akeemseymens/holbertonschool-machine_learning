#!/usr/bin/env python3
"""deep NN performing binary classififcation"""

import numpy as np


class DeepNeuralNetwork:
    """Deep Neural Network Class"""

    def __init__(self, nx, layers):
        """nx is number of input values"""
        if type(nx) is not (int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        """layers list reping num nodes in each layer"""
        if type(layers) is not (list) or len(layers) <= 0:
            raise TypeError("layers must be a list of positive integers")
        self.__L = len(layers)
        self.__nx = nx
        self.__cache = {}
        self.__weights = {}
        for i_lyr in range(self.L):
            mWts = "W" + str(i_lyr + 1)
            mB = "b" + str(i_lyr + 1)
            if type(layers[i_lyr]) is not (int) or layers[i_lyr] < 1:
                raise TypeError("layers must be a list of positive integers")
            self.__weights[mB] = np.zeros((layers[i_lyr], 1))
            if i_lyr == 0:
                self.__weights[mWts] = (np.random.randn(layers[i_lyr], nx)
                                        * np.sqrt(2 / nx))
            else:
                self.__weights[mWts] = (np.random.randn(layers[i_lyr],
                                        layers[i_lyr - 1])
                                        * np.sqrt(2 / layers[i_lyr - 1]))

    @property
    def L(self):
        """returns length of layers"""
        return self.__L

    @property
    def nx(self):
        """returns number of input values"""
        return self.__nx

    @property
    def cache(self):
        """returns dictionary with values of network"""
        return self.__cache

    @property
    def weights(self):
        """return dictionary w/ weights & bias of network"""
        return self.__weights

    def forward_prop(self, X):
        """Calculates forward propagation of NN"""
        self.__cache["A0"] = X
        for layer in range(self.__L):
            idx = layer + 1
            A_prev = self.__cache["A" + str(layer)]
            W_curr = self.__weights["W" + str(idx)]
            b_curr = self.__weights["b" + str(idx)]
            Z = np.matmul(W_curr, A_prev) + b_curr
            self.__cache["A" + str(idx)] = 1 / (1 + np.exp(-Z))
        return self.__cache["A" + str(self.__L)], self.__cache

    def cost(self, Y, A):
        """Calculates cost of model using logistic regression"""
        return -(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)).mean()

    def evaluate(self, X, Y):
        M = self.forward_prop(X)[0]
        return M.round().astype(int), self.cost(Y, M)

    def gradient_descent(self, Y, cache, alpha=0.05):
        """calculate one pass of gradient descent
           on the neural network"""

        m = Y.shape[1]
        last_key = 'A' + str(self.L)
        dZ = cache[last_key] - Y
        for i in range(self.L, 0, -1):
            key = 'A' + str(i - 1)
            A = cache[key]
            dW = (1 / m) * np.matmul(dZ, A.T)
            db1 = (np.sum(dZ, axis=1, keepdims=True) / m)
            W = self.__weights['W' + str(i)]
            dZ = np.matmul(W.T, dZ) * (A * (1-A))
            self.__weights['W' + str(i)] -= alpha * dW
            self.__weights['b' + str(i)] -= alpha * db1