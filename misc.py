import tensorflow as tf


def leaky_relu(x):
    """
        Leaky relu implementation inspired from keras
    """
    negative_part = tf.nn.relu(-x)
    x = tf.nn.relu(x)
    x -= 0.2 * negative_part
    return x
