# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
import tensorflow as tf

KH = 5
KW = 5


def batch_norm(x, is_training, epsilon=1e-5, decay=0.9, scope="batch_norm"):
    """
    Batch norm
    :param x:
    :param is_training:
    :param epsilon:
    :param decay:
    :param scope:
    :return:
    """
    return tf.contrib.layers.batch_norm(x, decay=decay, updates_collections=None, epsilon=epsilon, scale=True,
                                        is_training=is_training, scope=scope)


def conv2d(x, output_filters, kh=KH, kw=KW, sh=2, sw=2, stddev=0.02, scope="conv2d"):
    """
    Convolutional layer
    :param x:
    :param output_filters:
    :param kh:
    :param kw:
    :param sh:
    :param sw:
    :param stddev:
    :param scope:
    :return:
    """
    with tf.variable_scope(scope):
        shape = x.shape
        W = tf.get_variable('W', [kh, kw, shape[-1], output_filters],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))

        Wconv = tf.nn.conv2d(x, W, strides=[1, sh, sw, 1], padding='SAME')

        biases = tf.get_variable('b', [output_filters], initializer=tf.constant_initializer(0.0))

        Wconv_plus_b = tf.reshape(tf.nn.bias_add(Wconv, biases), Wconv.get_shape())

        return Wconv_plus_b


def deconv2d(x, output_shape, kh=KH, kw=KW, sh=2, sw=2, stddev=0.02, scope="deconv2d"):
    """
    De convolutional layer
    :param x:
    :param output_shape:
    :param kh:
    :param kw:
    :param sh:
    :param sw:
    :param stddev:
    :param scope:
    :return:
    """
    with tf.variable_scope(scope):
        input_shape = x.shape
        W = tf.get_variable('W', [kh, kw, output_shape[-1], input_shape[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))

        deconv = tf.nn.conv2d_transpose(x, W, output_shape=output_shape, strides=[1, sh, sw, 1])

        biases = tf.get_variable('b', [output_shape[-1]], initializer=tf.constant_initializer(0.0))

        deconv_plus_b = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        return deconv_plus_b


def lrelu(x, leak=0.2):
    """
    Leaky RELU
    :param x:
    :param leak:
    :return:
    """
    return tf.maximum(x, leak * x)


def fc(x, output_size, stddev=0.02, scope="fc"):
    """
    Fully connection layer
    :param x:
    :param output_size:
    :param stddev:
    :param scope:
    :return:
    """
    with tf.variable_scope(scope):
        shape = x.get_shape().as_list()
        W = tf.get_variable("W", [shape[1], output_size], tf.float32, tf.random_normal_initializer(stddev=stddev))
        b = tf.get_variable("b", [output_size], initializer=tf.constant_initializer(0.0))

        return tf.matmul(x, W) + b