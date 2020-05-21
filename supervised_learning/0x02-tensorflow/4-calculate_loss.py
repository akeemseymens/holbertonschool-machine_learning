#!/usr/bin/env python3
""" Calculate softmax cross-entropy loss of a prediction"""

import tensorflow as tf


def calculate_loss(y, y_pred):
    """Return a tensor containing the loss of the prediction."""
    cost = tf.losses.softmax_cross_entropy(logits=y_pred, onehot_labels=y)
    return cost
