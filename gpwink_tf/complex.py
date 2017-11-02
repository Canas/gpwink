import numpy as np
import tensorflow as tf

from gpwink_tf import GLOBAL_DTYPE
from gpwink_tf.utils import augment


class ComplexGaussian:
    def __init__(self, n_values, mean=None, cov=None, pcov=None):
        self._n_values = n_values
        self._mean = mean
        self._cov = cov
        self._pcov = pcov

    @classmethod
    def from_params(cls, mean, alpha, beta, gamma, delta):
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
        return augment(self._mean)

    def augmented_covariance(self):
        return augment(self._cov, self._pcov)
