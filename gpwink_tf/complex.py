# -*- coding: utf-8 -*-
"""
gpwink_tf.complex
~~~~~~~~~~~~~~~~~
This module provides parameterized complex gaussian definitions.
The advantage of this class is to provide initialization via
arbitrary parameters that ensure semi positive definite covariances so
their Cholesky decomposition can be calculated always.
"""

import numpy as np
import tensorflow as tf

from gpwink_tf import GLOBAL_DTYPE
from gpwink_tf.utils import augment


class ComplexGaussian:
    """Complex Gaussian model"""

    def __init__(self, n_values, mean=None, cov=None, pcov=None):
        """Create a raw Complex Multivariate Gaussian model.

        This method initializes a Complex Gaussian class with known
        mean, covariance and pseudo-covariances. To ensure Cholesky
        decomposition during optimization use the from_params class
        method.

        :param n_values: number of variables
        :param mean: mean array of the variables
        :param cov: covariance matrix of the variables
        :param pcov: pseudo-covariance matrix of the variables
        """
        self._n_values = n_values
        self._mean = mean
        self._cov = cov
        self._pcov = pcov
        self._a_mean = None
        self._a_cov = None

    @classmethod
    def from_params(cls, mean, alpha, beta, gamma, delta):
        """Create a Complex Multivariate Gaussian Model via parameterization.

        This method uses a parameterized definition of the variance and
        covariance to create the matrices for the Complex Gaussian object.
        We can define the complex gaussian variable z via the combination
        of three real gaussian vairables x, y, w, with mean 0 and cov 1

        z_real = alpha * x + beta  * y
        z_imag = gamma * y * delta * w

        Because of this formulation, the variances and covariances take
        the following form:

        cov(z) = E(z * z^H) = alpha * alpha^T + beta * beta^T +
                              gamma * gamma^T + delta * delta^T +
                              j * (gamma * beta^T + beta * gamma^T)

        pcov(z) = E(z * z)  = alpha * alpha^T + delta * delta^T +
                              j * (gamma * beta^T + beta * gamma^T)

        This ensures that we can generate covariances stay semi positive
        definite during the optimization step without crashing.

        :param mean: mean array of the variables
        :param alpha: positive Variable
        :param beta: positive Variable
        :param gamma: positive Variable
        :param delta: positive Variable
        :return: ComplexGaussian object with mean, cov, and pcov defined
        """
        n_values = mean.get_shape()[0]

        mean = tf.cast(mean, dtype=GLOBAL_DTYPE)

        cov = tf.complex(
            alpha @ tf.transpose(alpha) + beta @ tf.transpose(beta) + \
                gamma @ tf.transpose(gamma) + delta @ tf.transpose(delta),
            gamma @ tf.transpose(beta) - beta @ tf.transpose(gamma)
        )
        pcov = tf.complex(
            alpha @ tf.transpose(alpha) + beta @ tf.transpose(beta) - \
                gamma @ tf.transpose(gamma) - delta @ tf.transpose(delta),
            gamma @ tf.transpose(beta) + beta @ tf.transpose(gamma)
        )

        cov = tf.cast(cov, GLOBAL_DTYPE)
        pcov = tf.cast(pcov, GLOBAL_DTYPE)

        return ComplexGaussian(n_values, mean=mean, cov=cov, pcov=pcov)

    def augmented_mean(self):
        """Augmented mean array"""
        if self._a_mean is None:
            self._a_mean = augment(self._mean)
        return self._a_mean

    def augmented_covariance(self):
        """Augmented covariance matrix"""
        if self._a_cov is None:
            self._a_cov = augment(self._cov, self._pcov)
        return self._a_cov
