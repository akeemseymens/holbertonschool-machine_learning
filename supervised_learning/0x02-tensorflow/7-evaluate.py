#!/usr/bin/env python3
"""Evaluates the output of a neural network"""

import tensorflow as tf


def evaluate(X, Y, save_path):
    """Evaluate the output of a neural network"""
    with tf.Session() as sess:
        save_file = tf.train.import_meta_graph('save_path/model.ckpt.meta')
        save_file.restore(sess, save_path)
        var_names = ['x', 'y', 'y_pred', 'accuracy', 'loss']
        for var_name in var_names:
            globals()[var_name] = tf.get_collection(var_name)[0]
        y_pred = sess.run(globals()['y_pred'], feed_dict={x: X, y: Y})
        loss = sess.run(globals()['loss'], feed_dict={x: X, y: Y})
        acc = sess.run(globals()['accuracy'], feed_dict={x: X, y: Y})
    return y_pred, acc, loss
