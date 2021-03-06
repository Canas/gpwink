# -*- coding: utf-8 -*-
"""
gpwink_tf.utils
~~~~~~~~~~~~~~~~~~~~~~~~
This module provides utility functions used across the package.
"""

import numpy as np
import tensorflow as tf


def augment(cov, pcov=None):
    """Augments complex covariance matrices.

    Given the complex variable covariance C and pseudo-covariance P,
    the method returns the matrix: [C  P; P* C*] where (*) denotes
    the conjugate of a matrix.

    :param cov: covariance matrix (N x N)
    :param pcov: pseudo-covariance matrix (N x N)
    :return: (N x 2N) matrix if pcov=None; else (2N x 2N) matrix
    """
    if pcov is None:
        upper = cov
        lower = tf.conj(cov)
    else:
        upper = tf.concat((cov, pcov), axis=1)
        lower = tf.concat((tf.conj(pcov), tf.conj(cov)), axis=1)

    return tf.concat((upper, lower), axis=0)


def my_inner(a, b):
    """Inner product between two matrices. """
    return tf.reduce_sum(a * b, axis=(0, 1))


def np_decaying_square_exponential(space, alpha=1, gamma=1/2, sigma=1):
    """Sample from a decaying square exponential function. """
    base = np.zeros(len(space))
    first_space = space[:, None] + base
    second_space = space[None, :] + base
    outer_difference = first_space - second_space
    return np.exp(-alpha * first_space**2 - alpha * second_space**2 + -gamma * outer_difference**2)
