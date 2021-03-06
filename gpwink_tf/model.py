# -*- coding: utf-8 -*-
"""
gpwink_tf.model
~~~~~~~~~~~~~~~
This module contains the main Gaussian Process with Non-parametric Kernels model.
The model provides methods to calculate expectation, moments, variance and the
Evidence Lower BOund (ELBO) which must be minimized in the optimization step.
"""
import math

import tensorflow as tf

from gpwink_tf import GLOBAL_DTYPE
from gpwink_tf.integral import integrate_window_kernel_window, integrate_window
from gpwink_tf.kullback_leiblers import kl_complex_gaussian
from gpwink_tf.utils import my_inner

PI = tf.constant(math.pi, dtype=GLOBAL_DTYPE)


class GPModel:
    """Base class for Gaussian Process Model. """

    def __init__(self, t_obs, y_obs, kernel):
        """Gaussian Process model definition.

        :param t_obs: observed data indices
        :param y_obs: observed data values
        """
        self.n = t_obs.get_shape()[0].value
        self.t_obs = t_obs
        self.y_obs = y_obs
        self.kernel = kernel


class GPWiNK(GPModel):
    def __init__(self, t_obs, y_obs, filter_kernel, filter_inducing,
                 noise_inducing, filter_variational, noise_variational):
        """ Vanilla GPWiNK implementation.

        :param t_obs: observed data indices
        :param y_obs: observed data values
        :param filter_kernel: kernel function for filter term
        :param filter_inducing: inducing variable for filter term
        :param noise_inducing: inducing variable for noise term
        :param filter_variational: variational variable for filter term
        :param noise_variational: variational variable for noise term
        """
        super().__init__(t_obs, y_obs, filter_kernel)
        self.filter_inducing = filter_inducing
        self.noise_inducing = noise_inducing
        self.filter_variational = filter_variational
        self.noise_variational = noise_variational

        # may change to Variable
        self.sigma_x = tf.constant(1., dtype=GLOBAL_DTYPE)
        self.sigma_y = tf.constant(1., dtype=GLOBAL_DTYPE)

        # properties
        self._au_inverse = None
        self._av_inverse = None
        self._au_inverse_times_qu_mean = None
        self._av_inverse_times_qv_mean = None
        self._mu = None
        self._mv = None

        # this is doing the same thing for now until covariance augmentation
        # is not applied if using non-complex dtypes
        if not GLOBAL_DTYPE == tf.complex64:
            self._eps_filter = 1e-10 * tf.eye(2 * self.filter_inducing.n_values,
                                              dtype=GLOBAL_DTYPE)
            self._eps_noise = 1e-10 * tf.eye(2 * self.noise_inducing.n_values,
                                             dtype=GLOBAL_DTYPE)
        else:
            self._eps_filter = 1e-10 * tf.eye(2 * self.filter_inducing.n_values,
                                              dtype=GLOBAL_DTYPE)
            self._eps_noise = 1e-10 * tf.eye(2 * self.noise_inducing.n_values,
                                             dtype=GLOBAL_DTYPE)

    def au_inverse(self):
        """Inverse of the filter inducing augmented covariance matrix. """
        if self._au_inverse is None:
            epsilon = 1e-10 * tf.eye(2 * self.filter_inducing.n_values,
                                     dtype=GLOBAL_DTYPE)
            self._au_inverse =  tf.matrix_inverse(
                self.filter_inducing.augmented_covariance() + self._eps_filter,
                name="au_inverse"
            )
        return self._au_inverse

    def av_inverse(self):
        """Inverse of the noise inducing augmented covariance matrix. """
        if self._av_inverse is None:
            epsilon = 1e-10 * tf.eye(2 * self.noise_inducing.n_values,
                                     dtype=GLOBAL_DTYPE)
            self._av_inverse = tf.matrix_inverse(
                self.noise_inducing.augmented_covariance() + self._eps_noise,
                name="av_inverse"
            )
        return self._av_inverse

    def au_inverse_times_qu_mean(self):
        """Inverse of the filter inducing augmented covariance matrix times the
        augmented mean of the filter variational approximation . """
        if self._au_inverse_times_qu_mean is None:
            epsilon = 1e-10 * tf.eye(2 * self.filter_inducing.n_values,
                                     dtype=GLOBAL_DTYPE)
            self._au_inverse_times_qu_mean = tf.matrix_solve(
                self.filter_inducing.augmented_covariance() + self._eps_filter,
                self.filter_variational.augmented_mean(),
                name="au_inverse_time_qu_mean"
            )
        return self._au_inverse_times_qu_mean

    def av_inverse_times_qv_mean(self):
        """Inverse of the noise inducing augmented covariance matrix times the
        augmented mean of the noise variational approximation. """
        if self._av_inverse_times_qv_mean is None:
            epsilon = 1e-10 * tf.eye(2 * self.noise_inducing.n_values,
                                     dtype=GLOBAL_DTYPE)
            self._av_inverse_times_qv_mean = tf.matrix_solve(
                self.noise_inducing.augmented_covariance() + self._eps_noise,
                self.noise_variational.augmented_mean(),
                name="au_inverse_time_qu_mean"
            )
        return self._av_inverse_times_qv_mean

    def mu(self):
        """Mu matrix defined to avoid redundant calculations. """
        if self._mu is None:
            self._mu = self.au_inverse() - self.au_inverse_times_qu_mean() @ \
                   tf.transpose(self.au_inverse_times_qu_mean())
        return self._mu

    def mv(self):
        """Mv matrix defined to avoid redundant calculations. """
        if self._mv is None:
            self._mv = self.av_inverse() - self.av_inverse_times_qv_mean() @ \
                   tf.transpose(self.av_inverse_times_qv_mean())
        return self._mv

    def mean(self, t_new):
        """Mean of the process given an input. """
        with tf.variable_scope('first_moment'):
            auh = self.filter_inducing.augmented_interdomain_covariance()
            axv = self.noise_inducing.augmented_interdomain_covariance()
            m_linear = integrate_window(auh, axv, scale_left=-1, shift_left=t_new)
            return tf.transpose(self.au_inverse_times_qu_mean()) @ \
                m_linear @ self.av_inverse_times_qv_mean()

    def second_moment(self, t_new):
        """Second moment of the process given an input. """
        with tf.variable_scope('second_moment'):
            auh = self.filter_inducing.augmented_interdomain_covariance()
            avx = self.noise_inducing.augmented_interdomain_covariance()
            m_linear = integrate_window(auh, avx, scale_left=-1, shift_left=t_new)

            term1 = self.kernel.integrate_along_diagonal(scale=-1, shift=t_new)
            term2 = integrate_window_kernel_window(
                avx, self.kernel, avx, scale_mid=(-1, -1),
                shift_mid=(t_new, t_new)
            )
            term3 = integrate_window(auh, auh, scale_left=-1, shift_left=t_new,
                                     scale_right=-1, shift_right=t_new)
            term4 = m_linear
            return tf.subtract(
                tf.subtract(
                    term1,
                    my_inner(self.mv(), term2)
                ),
                tf.add(
                    my_inner(self.mu(), term3),
                    my_inner(self.mu() @ term4, term4 @ self.mv())
                )
            )

    def elbo(self):
        """Evidence Lower Bound of the process given the observations. """
        with tf.variable_scope('elbo'):
            constant_term = - 0.5 * self.n * tf.log(2. * PI * self.sigma_y**2) \
                       - 0.5 * (1 / self.sigma_y**2) * tf.reduce_sum(self.y_obs**2)

            linear_term = 0.
            quadratic_term = 0.

            # exec some properties once to avoid call errors during tf.map_fn
            # maybe there is a more elegant solution but for now this fix works
            mu = self.mu()
            mv = self.mv()
            auh = self.filter_inducing.augmented_interdomain_covariance()
            avx = self.noise_inducing.augmented_interdomain_covariance()
            m_linear = integrate_window(auh, avx, scale_left=-1, shift_left=self.t_obs[0])

            linear_term = tf.reduce_sum(
                    -2.0 * self.y_obs * tf.map_fn(self.mean, self.t_obs, name='linear_term')
                )

            quadratic_term = tf.reduce_sum(
                    tf.map_fn(self.second_moment, self.t_obs, name='quadratic_term')
                )

            kl_term = tf.add(
                    kl_complex_gaussian(
                        self.filter_variational.augmented_mean(),
                        self.filter_variational.augmented_covariance(),
                        0.,
                        self.filter_inducing.augmented_covariance(),
                        name='filter_kl'
                    ),
                    kl_complex_gaussian(
                        self.noise_variational.augmented_mean(),
                        self.noise_variational.augmented_covariance(),
                        0.,
                        self.noise_inducing.augmented_covariance(),
                        name='noise_kl'
                    ),
                    name='kl_term'
                )

            return constant_term + linear_term + \
                quadratic_term + kl_term
