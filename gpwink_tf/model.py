import math

import tensorflow as tf

from gpwink2 import GLOBAL_DTYPE
from gpwink2.integral import integrate_window_kernel_window, integrate_window
from gpwink2.kullback_leiblers import kl_complex_gaussian
from gpwink2.utils import my_inner

PI = tf.constant(math.pi, dtype=GLOBAL_DTYPE)


class GPModel:
    def __init__(self, t_obs, y_obs, kernel):
        """ Gaussian Process model basis

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

    def au_inverse(self):
        epsilon = 1e-10 * tf.eye(2 * self.filter_inducing.n_values,
                                 dtype=GLOBAL_DTYPE)
        return tf.matrix_inverse(
            self.filter_inducing.augmented_covariance() + epsilon,
            name="au_inverse"
        )

    def av_inverse(self):
        epsilon = 1e-10 * tf.eye(2 * self.noise_inducing.n_values,
                                 dtype=GLOBAL_DTYPE)
        return tf.matrix_inverse(
            self.noise_inducing.augmented_covariance() + epsilon,
            name="av_inverse"
        )

    def au_inverse_times_qu_mean(self):
        epsilon = 1e-10 * tf.eye(2 * self.filter_inducing.n_values,
                                 dtype=GLOBAL_DTYPE)
        return tf.matrix_solve(
            self.filter_inducing.augmented_covariance() + epsilon,
            self.filter_variational.augmented_mean(),
            name="au_inverse_time_qu_mean"
        )

    def av_inverse_times_qv_mean(self):
        epsilon = 1e-10 * tf.eye(2 * self.noise_inducing.n_values,
                                 dtype=GLOBAL_DTYPE)
        return tf.matrix_solve(
            self.noise_inducing.augmented_covariance() + epsilon,
            self.noise_variational.augmented_mean(),
            name="au_inverse_time_qu_mean"
        )

    def mu(self):
        return self.au_inverse() - self.au_inverse_times_qu_mean() @ \
               tf.transpose(self.au_inverse_times_qu_mean())

    def mv(self):
        return self.av_inverse() - self.av_inverse_times_qv_mean() @ \
               tf.transpose(self.av_inverse_times_qv_mean())

    def mean(self, t_new):
        auh = self.filter_inducing.augmented_interdomain_covariance()
        axv = self.noise_inducing.augmented_interdomain_covariance()
        m_linear = integrate_window(auh, axv, scale_left=-1, shift_left=t_new)
        return tf.transpose(self.au_inverse_times_qu_mean()) @ \
               m_linear @ self.av_inverse_times_qv_mean()

    def second_moment(self, t_new):
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

    def build_elbo(self):
        def build_constant_term():
            return - 0.5 * self.n * tf.log(2. * PI * self.sigma_y**2) \
                   - 0.5 * (1 / self.sigma_y**2) * tf.reduce_sum(self.y_obs**2)

        def build_linear_term():
            return tf.reduce_sum(
                -2.0 * self.y_obs * tf.map_fn(self.mean, self.t_obs)
            )

        def build_quadratic_term():
            return tf.reduce_sum(
                tf.map_fn(self.second_moment, self.t_obs)
            )

        def build_kl_divergence_term():
            return tf.add(
                kl_complex_gaussian(
                    self.filter_variational.augmented_mean(),
                    self.filter_variational.augmented_covariance(),
                    0.,
                    self.filter_inducing.augmented_covariance()
                ),
                kl_complex_gaussian(
                    self.noise_variational.augmented_mean(),
                    self.noise_variational.augmented_covariance(),
                    0.,
                    self.noise_inducing.augmented_covariance()
                )
            )

        return build_constant_term() + build_linear_term() + \
               build_quadratic_term() + build_kl_divergence_term()
