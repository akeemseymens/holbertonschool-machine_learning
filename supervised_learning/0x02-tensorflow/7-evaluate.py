#!/usr/bin/env python3
"""Evaluates the output of a neural network"""

import tensorflow as tf


def evaluate(X, Y, save_path):
    """Evaluate the output of a neural network"""
    with tf.Session() as sess:
        save_file = tf.train.import_meta_graph(save_path + '.meta')
        save_file.restore(sess, save_path)
        var_names = ['x', 'y', 'pred', 'accuracy', 'loss']
        for var_name in var_names:
            globals()[var_name] = tf.get_collection(var_name)[0]
        pred = sess.run(globals()['pred'], feed_dict={x: X, y: Y})
        loss = sess.run(globals()['loss'], feed_dict={x: X, y: Y})
        acc = sess.run(globals()['accuracy'], feed_dict={x: X, y: Y})
    return pred, acc, loss
