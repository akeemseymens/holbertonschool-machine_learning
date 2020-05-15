#!/usr/bin/env python3
"""Deep NN performing binary classififcation."""

import numpy as np
import matplotlib.pyplot as plt
import pickle

class DeepNeuralNetwork:
    """Deep Neural Network Class."""

    def __init__(self, nx, layers):
        """Nx is number of input values."""
        if type(nx) is not (int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        #Layers list reping num nodes in each layer
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
        """Return length of layers."""
        return self.__L

    @property
    def nx(self):
        """Return number of input values."""
        return self.__nx

    @property
    def cache(self):
        """Return dictionary with values of network."""
        return self.__cache

    @property
    def weights(self):
        """Return dictionary w/ weights & bias of network."""
        return self.__weights

    def forward_prop(self, X):
        """forward propagation function"""
        self.__cache['A0'] = X
        for i in range(self.L):
            Zi = np.matmul(
                self.weights['W' + str(i + 1)], self.cache['A' + str(i)]
            ) + self.weights['b' + str(i + 1)]
            # output layer: use softmax activation function
            if i == self.L - 1:
                self.__cache['A' + str(i + 1)] = self.softmax(Zi)
            else:
                self.__cache['A' + str(i + 1)] = self.sigmoid(Zi)
        return self.cache['A' + str(i + 1)], self.cache

    def cost(self, Y, A):
        """define the cost function"""
        m = Y.shape[1]
        return -np.sum(Y * np.log(A)) / m()

    def evaluate(self, X, Y):
        A, cache = self.forward_prop(X)
        cost = self.cost(Y, A)
        M = np.max(A, axis=0)
        return np.where(A == M, 1, 0), costt(Y, M)

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Calculate one pass of gradient descent."""
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

    def sigmoid(self, Y):
        """define the sigmoid activation function"""
        return 1 / (1 + np.exp(-Y))

    def softmax(self, Y):
        """define the softmax activation function"""
        return np.exp(Y) / (np.sum(np.exp(Y), axis=0, keepdims=True))

    def train(self, X, Y, iterations=5000, alpha=0.05,
                verbose=True, graph=True, step=100):
        """Train the neural network by updating the private attributes."""
        if type(iterations) != int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) != float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose:
            if type(step) != int:
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        if graph:
            if type(step) != int:
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        x = []
        y = []
        for iteration in range(iterations):
            output = self.forward_prop(X)
            self.gradient_descent(Y, self.cache, alpha)
            if verbose and iteration % step == 0:
                print("Cost after {} iterations: {}"
                        .format(iteration, self.cost(Y, output[0])))
                x.append(iteration)
                y.append(self.cost(Y, output[0]))
            if graph and iteration % step == 0:
                x.append(iteration)
                y.append(self.cost(Y, output[0]))

        if verbose:
            print("Cost after {} iterations: {}"
                    .format(iteration, self.cost(Y, output[0])))
        if graph:
            x.append(iteration)
            y.append(self.cost(Y, output[0]))
            plt.plot(x, y, 'b-')
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
        return (self.evaluate(X, Y))

    def save(self, filename):
        """Save the instance object to a file in pickle format."""
        if not filename.endswith('.pkl'):
            filename += '.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """Load a pickled DeepNeuralNetwork object."""
        if filename is None:
            return None
        try:
            with open(filename, 'rb') as f:
                return (pickle.load(f))
        except FileNotFoundError:
            return None
