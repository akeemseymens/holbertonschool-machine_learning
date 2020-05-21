#!/usr/bin/env python3
"""
function that calculate the accuracy of a prediction
"""
import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """return a tensor containing the decimal
       accuracy of the prediction"""
    correct_pred = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return accuracy
