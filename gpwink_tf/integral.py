# -*- coding: utf-8 -*-
"""
gpwink.integrate
~~~~~~~~~~~~~~~~

This module provides integral shortcuts for kernels and kernel windows.
"""

import numpy as np
import tensorflow as tf

from gpwink2 import GLOBAL_DTYPE
from gpwink2.integrable import GaussianSquareExponentialWindow


def integrate_window(left_window, right_window=None, scale_left=1,
                     scale_right=1, shift_left=0, shift_right=0,
                     conjugate_right=False):
    """Solve integral for window-window combination.

    Evaluates the integral of same-window multiplication with respect to one
    variable: ∫ W(tau) W(tau).T dtau
    :return:
    """
    if right_window is None:
        right_window = left_window

    left_window = left_window.scale_and_shift(scale_left, shift_left)
    left_sigma, left_t2, left_t, left_1 = left_window.get_all_params()

    right_window = right_window.scale_and_shift(scale_right, shift_right)
    if conjugate_right:
        right_window = right_window.conjugate()
    right_sigma, right_s2, right_s, right_1 = right_window.get_all_params()

    sigma = left_sigma * tf.transpose(right_sigma)
    ct2 = left_t2 + tf.transpose(right_s2)
    ct = left_t + tf.transpose(right_s)
    c1 = left_1 + tf.transpose(right_1)

    return sigma**2 * tf.sqrt(np.pi / -ct2) * tf.exp(ct**2 / (4 * -ct2) + c1)


def integrate_kernel(kernel, scale=(1, 1), shift=(0, 0)):
    """Solve integral for kernel single argument.

    Evaluates the integral of a kernel with respect to one variable:
    ∫ K(tau, tau') dtau
    """
    kernel = kernel.scale_and_shift(scale, shift)
    sigma, ct2, cs2, cts, ct, cs, c1 = kernel.get_all_params()

    ct2 += cs2 + cts  # why?
    ct += cs  # why?

    return sigma**2 * tf.sqrt(np.pi/-ct2) * tf.exp(ct**2 / (4 * -ct2) + c1)


def integrate_kernel_window(kernel, window, scale_kernel, shift_kernel,
                            scale_window, shift_window, conjugate_right=False):
    """Solve integral for kernel-window combination.

    Evaluates the case where the argument is a window in t and s, but only is
    integrated with respect to s; output is window in t:

    ∫ K(tau, tau') W(tau) dtau

    :param kernel:
    :param window:
    :param scale_kernel:
    :param shift_kernel:
    :param scale_window:
    :param shift_window:
    :param conjugate_right:
    :return:
    """
    window = window.scale_and_shift(scale_window, shift_window)
    if conjugate_right:
        window = window.conjugate()
    win_sigma, win_s2, win_s, win_1 = window.get_all_params()

    kernel = kernel.scale_and_shift(scale_kernel, shift_kernel)
    kern_sigma, kern_t2, kern_s2, kern_ts, kern_t, kern_s, kern_1 = \
        kernel.get_all_params()

    sigma = kern_sigma * win_sigma
    ct2 = kern_t2 + tf.zeros_like(sigma)
    cs2 = kern_s2 + win_s2 + tf.zeros_like(sigma)
    cts = kern_ts + tf.zeros_like(sigma)
    ct = kern_t + tf.zeros_like(sigma)
    cs = kern_s + win_s + tf.zeros_like(sigma)
    c1 = kern_1 + win_1 + tf.zeros_like(sigma)

    new_ct2 = ct2 + cts**2 / (4 * cs2)
    new_ct = ct + cts * cs / (2 * cs2)
    new_c1 = cs**2 / (4 * cs2) + c1
    return GaussianSquareExponentialWindow(
        sigma=tf.cast(tf.cast(sigma, tf.float64), GLOBAL_DTYPE),
        ct2=tf.cast(tf.cast(new_ct2, tf.float64), GLOBAL_DTYPE),
        ct=new_ct,
        c1=new_c1
    )


def integrate_window_kernel_window(left_window, kernel, right_window,
                                   scale_left=1, scale_right=1,
                                   scale_mid=(1, 1), shift_left=0,
                                   shift_right=0, shift_mid=(0, 0),
                                   conjugate_right=False):
    """Solve integral for window-kernel-window combination.

    ∫ W(tau) K(tau, tau') W(tau') dtau dtau'
    """

    left_window = left_window.scale_and_shift(scale_left, shift_left)
    left_sigma, left_t2, left_t, left_1 = left_window.get_all_params()

    right_window = right_window.scale_and_shift(scale_right, shift_right)
    if conjugate_right:
        right_window = right_window.conjugate()
    right_sigma, right_s2, right_s, right_1 = right_window.get_all_params()

    kernel = kernel.scale_and_shift(scale_mid, shift_mid)
    mid_sigma, mid_t2, mid_s2, mid_ts, mid_t, mid_s, mid_1 = \
        kernel.get_all_params()

    sigma = left_sigma * mid_sigma * tf.transpose(right_sigma)

    # sigma rules the size
    ct2 = left_t2 + mid_t2 + tf.zeros_like(sigma)
    cs2 = mid_s2 + tf.transpose(right_s2) + tf.zeros_like(sigma)
    cts = mid_ts + tf.zeros_like(sigma)
    ct = left_t + mid_t + tf.zeros_like(sigma)
    cs = mid_s + tf.transpose(right_s) + tf.zeros_like(sigma)
    c1 = left_1 + mid_1 + tf.transpose(right_1) + tf.zeros_like(sigma)

    temp = 4 * ct2 * cs2 - cts ** 2

    return sigma ** 2 * 2 * np.pi / tf.sqrt(temp) * tf.exp(
        (-ct2 * ct ** 2 - cs2 * cs ** 2 + ct * cs * cts) / temp + c1
    )


def _scale_and_shift_window():
    pass


def _scale_and_shift_kernel():
    pass


def _conjugate():
    pass
