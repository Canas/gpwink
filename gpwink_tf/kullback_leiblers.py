import numpy as np
import tensorflow as tf

from gpwink_tf import GLOBAL_DTYPE


def kl_complex_gaussian(m1, c1, m2, c2, name='kl_complex_gaussian'):
    """ KL divergence for Complex Gaussian.

    Given two complex Gaussian distributions of the form
    p(x) = N(m1, c1)
    q(x) = N(m2, c2)
    where m and c are mean and covariances respectively, t

    :param m1: 1D array with mean of first model
    :param c1: 2D array with covariance of first model
    :param m2: 1D array with mean of second model
    :param c2: 2D array with covariance of second model
    :return: @TODO this
    """
    with tf.variable_scope(name):
        N = tf.cast(tf.shape(m1)[0], dtype=GLOBAL_DTYPE)
        # invc1_times_c2 = np.linalg.solve(c1, c2)
        invc1_times_c2 = tf.matrix_solve(c1, c2)
        # invc1_times_diff_m = np.linalg.solve(c1, m1 - m2)
        invc1_times_diff_m = tf.matrix_solve(c1, m1 - m2)
        L1 = tf.cholesky(c1, name='cholesky_variational_cov')
        # L1 = tf.py_func(np.linalg.cholesky, [c1], c1.dtype, name=f'cholesky_c1_{name}')
        L2 = tf.cholesky(c2, name='cholesky_inducing_cov')
        # L2 = tf.py_func(np.linalg.cholesky, [c2 + 0.1 * tf.eye(c2.get_shape()[0].value, dtype=GLOBAL_DTYPE)], c2.dtype, name=f'cholesky_c2_{name}')
        logdet1 = 2 * tf.reduce_sum(tf.log(tf.diag_part(L1)))
        logdet2 = 2 * tf.reduce_sum(tf.log(tf.diag_part(L2)))

        return 1. / 2. * (tf.trace(invc1_times_c2) + tf.transpose(m1 - m2) @
                          invc1_times_diff_m - N + logdet1 - logdet2)
