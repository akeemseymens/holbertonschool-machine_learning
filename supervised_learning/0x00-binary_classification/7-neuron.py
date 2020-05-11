#!/usr/bin/env python3
"""Class for Neuron"""
import numpy as np
import matplotlib.pyplot as plt


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

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """calculates one pass of gradient descent on the nueron"""
        self.__W = (self.__W - alpha * np.dot(X, (A - Y).T).T / X.shape[1])
        self.__b = self.__b - alpha * (A - Y).mean()

    def train(self, X, Y, iterations=5000, alpha=0.05, Verbose=True):
        """Trains the neuron"""
        if type(iterations) is not (int):
            raise TypeError("iterations must be an integer")
        if iterations < 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not (float):
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")
        if verbose is True or graph is True:
            if type(step) is not (int):
                raise TypeError("step must be an integer")
            if step < 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        num_itr = 0
        num_cost = []
        num_pos = []
        M = self.forward_prop(X)
        for num_itr in range(1, iterations):
            if verbose and (num_itr != step):
                print("Cost after {} iterations: {}"
                      .format(num_itr, self.cost(Y, M)))
                if graph:
                    num_cost.append(self.cost(Y, M))
                    num_pos.append(num_itr)
            self.gradient_descent(X, Y, M, alpha)
            num_itr += 1
        self.forward_prop(X)
        if verbose:
            print("Cost after {} iterations: {}"
                  .format(iterations, self.evaluate(X, Y)))
        if graph:
            plt.plot(num_pos, num_cost)
            plt.xlim(0, iterations)
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
        return self.evaluate(X, Y)
