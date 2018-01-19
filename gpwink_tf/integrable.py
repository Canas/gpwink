# -*- coding: utf-8 -*-
"""
gpwink_tf.integrable
~~~~~~~~~~~~~~~~~~~~
This module defines parametrized models that can be used by the
gpwink.integral module. Currently supported Integrable functions
are:

- Windows: Gaussian Square Exponential
- Kernels: Gaussian Decaying Square Exponenetial

Other window-kernel combinations may be explored in the future.
"""

import numpy as np
import tensorflow as tf

from gpwink_tf import GLOBAL_DTYPE
from gpwink_tf.utils import augment


class Integrable:
    """Generalized Integrable Functions model. """

    def __init__(self, **params):
        self.params = params

    def get_all_params(self):
        """Return the value of all parameters. """
        return {k: v for k, v in self.params.items()}


class GaussianSquareExponentialWindow(Integrable):
    """Gaussian Square Exponential function. """

    def __init__(self, **params):
        """Create a raw Gaussian Square Exponential Window.

        This method initializes a GaussianSquareExponentialWindow class 
        with known constant, linear and quadratic parameters. To initialize
        via the more familiar physical interpretation parameters use the 
        new_window class method.
        
        :param params: dictionary with scale factor sigma,
                       constant parameter c1, 
                       linear parameter ct,
                       quadractic parameter ct2
        """
        try:
            sigma = params['sigma']
            ct2   = params['ct2']
            ct    = params['ct']
            c1    = params['c1']
        except KeyError as e:
            raise KeyError(f'Missing parameter: {e}')

        super().__init__(**params)
        self.sigma = sigma
        self.ct2   = ct2
        self.ct    = ct
        self.c1    = c1

    def __call__(self, space):
        """Evaluate window on a set of time intervals. """
        return self.sigma * tf.exp(self.ct2 * space**2 +
                                   self.ct * space - self.c1)

    @classmethod
    def new_window(cls, sigma, gamma, centre, j_omega, j_phi):
        """Generates a Gaussian Exponential Window using the physical
        intepretation of parameters.

        ∫ σ exp( -1/(2*l^2) * (t - s)^2 + j(ωt + φ) )
        where σ, l > 0

        :param sigma: scaling parameter σ
        :param gamma: inverse of lengthscale parameter 1/(2*l^2)
        :param centre: centering parameter s
        :param j_omega: frequency parameter ω
        :param j_phi: phase parameter φ
        :return:
        """
        sigma   = tf.cast(sigma, dtype=GLOBAL_DTYPE)
        gamma   = tf.cast(gamma, dtype=tf.float64)
        centre  = tf.cast(centre, dtype=tf.float64)

        if not GLOBAL_DTYPE == tf.complex64:
            j_omega = tf.constant(0, dtype=GLOBAL_DTYPE)
            j_phi   = tf.constant(0, dtype=GLOBAL_DTYPE)
        else:
            j_omega = tf.cast(
                tf.complex(
                    tf.constant(0., dtype=tf.float64), 
                    tf.cast(j_omega, dtype=tf.float64)
                ), 
                dtype=tf.complex64
            )
            j_phi   = tf.cast(
                tf.complex(
                    tf.constant(0., dtype=tf.float64), 
                    tf.cast(j_phi, dtype=tf.float64)
                ),
                dtype=tf.complex64
            )

        c1 = tf.cast(-gamma * centre ** 2, dtype=GLOBAL_DTYPE) + j_phi
        ct = tf.cast(2 * gamma * centre, dtype=GLOBAL_DTYPE) + j_omega
        ct2 = -1 * tf.cast(gamma, dtype=GLOBAL_DTYPE)

        return GaussianSquareExponentialWindow(
            sigma=sigma, ct2=ct2, ct=ct, c1=c1
        )

    def get_all_params(self):
        """Return the value of all parameters. """
        return self.sigma, self.ct2, self.ct, self.c1

    def conjugate(self, inline=False):
        """Conjugate the ct and ct parameters of the window. """
        return GaussianSquareExponentialWindow(
            sigma=self.sigma,
            ct2=self.ct2,
            ct=tf.conj(self.ct),
            c1=tf.conj(self.c1)
        )

    def scale_and_shift(self, scale=1, shift=0, inline=False):
        """Scales and shifts the window. """
        ct2 = self.ct2 * scale ** 2
        ct = scale * (self.ct + 2 * self.ct2 * shift)
        c1 = tf.ones_like(shift, dtype=GLOBAL_DTYPE) * \
             (self.ct2 * shift ** 2 + self.ct * shift + self.c1)

        return GaussianSquareExponentialWindow(
            sigma=self.sigma,
            ct2=ct2,
            ct=ct,
            c1=c1
        )

    def augment(self):
        """Creates a new GaussianExponentialWindow using augmented attrs. """
        return GaussianSquareExponentialWindow(
            sigma=augment(self.sigma),
            ct2=augment(self.ct2),
            ct=augment(self.ct),
            c1=augment(self.c1)
        )


class GaussianSquareExponentialKernel(Integrable):
    """Gaussian Decaying Square Exponential kernel function. """

    def __init__(self, **params):
        """Create a raw Gaussian Square Exponential Kernel.

        This method initializes a GaussianSquareExponentialKernel class 
        with known constant, linear and quadratic parameters. To initialize
        via the more familiar physical interpretation parameters use the 
        new_kernel class method.
        
        :param params: dictionary with scale factor sigma,
                       constant parameter c1, 
                       linear parameters ct, cs and cts,
                       quadractic parameters ct2 and cs2
        """
        try:
            sigma = params['sigma']
            ct2   = params['ct2']
            cs2   = params['cs2']
            cts   = params['cts']
            ct    = params['ct']
            cs    = params['cs']
            c1    = params['c1']
        except KeyError as e:
            raise KeyError(f'Missing parameter: {e}')

        super().__init__(**params)
        self.sigma = sigma
        self.ct2   = ct2
        self.cs2   = cs2
        self.cts   = cts
        self.ct    = ct
        self.cs    = cs
        self.c1    = c1

    @classmethod
    def new_kernel(cls, sigma, gamma, alpha):
        """Generates a Gaussian Decaying Square Exponential Kernel using the physical
        intepretation of parameters.

        K(t1, t2) = σ^2 * exp(-α * t1^2) * exp(-γ * (t1 - t2)^2) * exp(-α * t2^2)
        where σ, α, γ > 0

        :param sigma: power output parameter σ
        :param gamma: parameter that controls the characteristic timescale 1/sqrt(γ)
        :param alpha: parameter that controls the temporal extend 1/sqrt(α)
        :return: GaussianSquareExponentialKernel object
        """
        sigma = tf.cast(sigma, dtype=GLOBAL_DTYPE)
        gamma = tf.cast(gamma, dtype=GLOBAL_DTYPE)
        alpha = tf.cast(alpha, dtype=GLOBAL_DTYPE)

        return GaussianSquareExponentialKernel(
            sigma=sigma,
            ct2=(-alpha - gamma),
            cs2=(-alpha - gamma),
            cts=2 * gamma,
            ct=0,
            cs=0,
            c1=0
        )

    @property
    def alpha(self):
        return -0.5 * (self.ct2 + self.cs2 + self.cts)

    def get_all_params(self):
        """Return the value of all parameters. """
        return self.sigma, self.ct2, self.cs2, self.cts, self.ct, self.cs, \
               self.c1

    def scale_and_shift(self, scale=(1, 1), shift=(0, 0), inline=False):
        """Scales and shifts the kernel.

        :param scale: tuple with scales for t and s
        :param shift: tuple with shifts for t and s
        :param inline: false to return new kernel, true to modify the object
        :return: kernel with new params or nothing, depending on inline param
        """
        ct2 = self.ct2 * scale[0]**2
        cs2 = self.cs2 * scale[1]**2
        cts = self.cts * scale[0] * scale[1]
        ct = 2 * self.ct2 * scale[0] * shift[0] + \
             self.cts * scale[0] * shift[1] + \
             self.ct * scale[0]
        cs = 2 * self.cs2 * scale[1] * shift[1] + \
             self.cts * scale[1] * shift[0] + \
             self.cs * scale[1]
        c1 = self.ct2 * shift[0]**2 + self.cs2 * shift[1]**2 + \
             self.cts * shift[0] * shift[1] + \
             self.ct * shift[0] + \
             self.cs * shift[1] + self.c1

        if inline:
            self.ct2 = ct2
            self.cs2 = cs2
            self.cts = cts
            self.ct = ct
            self.cs = cs
            self.c1 = c1
        else:
            return GaussianSquareExponentialKernel(
                sigma=self.sigma,
                ct2=ct2,
                cs2=cs2,
                cts=cts,
                ct=ct,
                cs=cs,
                c1=c1
            )

    def integrate_along_diagonal(self, scale=1, shift=0):
        # avoid unsupported type error if global dtype is complex
        alpha = tf.cast(self.alpha, tf.float64)

        # alpha needs to be strictly greater than zero
        # else integral may not be finite
        tf.where(
            tf.less_equal(alpha, tf.zeros_like(alpha)),
            tf.zeros_like(alpha),
            alpha
        )

        a0 = self.ct2 + self.cs2 + self.cts
        b0 = self.ct + self.cs
        c0 = self.c1

        a = a0 * tf.square(tf.cast(scale, dtype=GLOBAL_DTYPE))
        b = scale * (b0 + 2 * a0 * shift)
        c = tf.ones_like(shift, dtype=GLOBAL_DTYPE) * \
            (a0 * tf.square(shift) + b0 * shift + c0)
        return self.sigma * tf.sqrt(np.pi / -a) * tf.exp(b**2 / (4 * -a) + c)


def fix_variable(kernel, space):
    one_array = tf.ones_like(space, dtype=GLOBAL_DTYPE)
    return GaussianSquareExponentialWindow(
        sigma=kernel.sigma * one_array,
        ct2=kernel.ct2 * one_array,
        ct=kernel.ct + kernel.cts*space,
        c1=kernel.cs2*space**2 + kernel.cs*space + kernel.c1
    )
