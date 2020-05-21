#!/usr/bin/env python3
"""Create the forward propagation graph for the neural network."""

import tensorflow as tf

create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """return the prediction of the network in tensor form"""
    for layer, activation in zip(layer_sizes, activations):
        prev = create_layer(x, layer, activation)
        x = prev
    return prev
