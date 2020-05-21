
#!/usr/bin/env python3
"""
function that returns two placeholdres, x and y, for neural network
"""
import tensorflow as tf


def create_placeholders(nx, classes):
    """return two placeholders, x and y"""
    x = tf.placeholder(tf.float32, (None, nx), name='x')
    y = tf.placeholder(tf.float32, (None, classes), name='y')
    return (x, y)
