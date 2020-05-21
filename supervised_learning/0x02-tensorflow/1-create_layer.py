#!/usr/bin/env python3
"""
Return the tensor output of the layer
"""
import tensorflow as tf


def create_layer(prev, n, activation):
    """return the tensor ouput of the layer"""
    variance = tf.contrib.layers.variance_scaling_initializer(mode='FAN_AVG')
    layer = tf.layers.Dense(n, activation=activation,
                            kernel_initializer=variance, name='layer')
    return layer(prev)