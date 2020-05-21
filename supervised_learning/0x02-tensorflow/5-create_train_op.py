#!/usr/bin/env python3
"""Create the training operation for the network."""
import tensorflow as tf


def create_train_op(loss, alpha):
    """return an operation that trains the network using gradient descent"""
    a = tf.train.GradientDescentOptimizer(alpha)
    b = a.minimize(loss)
    return b
