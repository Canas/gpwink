# -*- coding: utf-8 -*-
"""
gpwink.integral
~~~~~~~~~~~~~~~~

This module provides integral shortcuts for kernels and kernel windows.
"""
from gpwink_tf.integral import (integrate_kernel_window, integrate_window,
                              integrate_window_kernel_window)
from gpwink_tf.utils import augment


class InducingVariable:
    def __init__(self, locations, interdomain_transform):
        self.n_values = locations.get_shape()[0].value
        self._locations = locations
        self._interdomain_transform = interdomain_transform
        self._cov = None
        self._pcov = None
        self._interdomain_cov = None
        self._a_cov = None
        self._a_interdomain_cov = None


class FilterInducingVariable(InducingVariable):
    def __init__(self, locations, interdomain_transform, kernel):
        super().__init__(locations, interdomain_transform)
        self._kernel = kernel

    def covariance(self):
        if self._cov is None:
            self._cov = integrate_window_kernel_window(
                self._interdomain_transform, self._kernel,
                self._interdomain_transform, conjugate_right=True,
                scale_left=-1, scale_mid=(1, 1), scale_right=-1,
                shift_left=0, shift_mid=(0, 0), shift_right=0
            )
        return self._cov

    def pseudo_covariance(self):
        if self._pcov is None:
            self._pcov = integrate_window_kernel_window(
                self._interdomain_transform, self._kernel,
                self._interdomain_transform, conjugate_right=False,
                scale_left=-1, scale_mid=(1, 1), scale_right=-1,
                shift_left=0, shift_mid=(0, 0), shift_right=0
            )
        return self._pcov

    def interdomain_covariance(self):
        if self._interdomain_cov is None:
            self._interdomain_cov = integrate_kernel_window(
                self._kernel, self._interdomain_transform, conjugate_right=True,
                scale_kernel=(1, -1), shift_kernel=(0, 0),
                scale_window=1, shift_window=0
            )
        return self._interdomain_cov

    def augmented_covariance(self):
        if self._a_cov is None:
            self._a_cov = augment(self.covariance(), self.pseudo_covariance())
        return self._a_cov

    def augmented_interdomain_covariance(self):
        if self._a_interdomain_cov is None:
            self._a_interdomain_cov = self.interdomain_covariance().augment()
        return self._a_interdomain_cov


class NoiseInducingVariable(InducingVariable):
    def __init__(self, locations, interdomain_transform):
        super().__init__(locations, interdomain_transform)

    def covariance(self):
        if self._cov is None:
            self._cov = integrate_window(
                self._interdomain_transform, self._interdomain_transform,
                conjugate_right=True, scale_right=1, scale_left=1,
                shift_left=0, shift_right=0
            )
        return self._cov

    def pseudo_covariance(self):
        if self._pcov is None:
            self._pcov = integrate_window(
                self._interdomain_transform, self._interdomain_transform,
                conjugate_right=False, scale_right=1, scale_left=1,
                shift_left=0, shift_right=0
            )
        return self._pcov

    def interdomain_covariance(self):
        if self._interdomain_cov is None:
            self._interdomain_cov = self._interdomain_transform.conjugate()
        return self._interdomain_cov

    def augmented_covariance(self):
        if self._a_cov is None:
            self._a_cov = augment(self.covariance(), self.pseudo_covariance())
        return self._a_cov

    def augmented_interdomain_covariance(self):
        if self._a_interdomain_cov is None:
            self._a_interdomain_cov = self.interdomain_covariance().augment()
        return self._a_interdomain_cov
