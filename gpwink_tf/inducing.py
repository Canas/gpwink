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


class FilterInducingVariable(InducingVariable):
    def __init__(self, locations, interdomain_transform, kernel):
        super().__init__(locations, interdomain_transform)
        self._kernel = kernel

    def covariance(self):
        return integrate_window_kernel_window(
            self._interdomain_transform, self._kernel,
            self._interdomain_transform, conjugate_right=True,
            scale_left=-1, scale_mid=(1, 1), scale_right=-1,
            shift_left=0, shift_mid=(0, 0), shift_right=0
        )

    def pseudo_covariance(self):
        return integrate_window_kernel_window(
            self._interdomain_transform, self._kernel,
            self._interdomain_transform, conjugate_right=False,
            scale_left=-1, scale_mid=(1, 1), scale_right=-1,
            shift_left=0, shift_mid=(0, 0), shift_right=0
        )

    def interdomain_covariance(self):
        return integrate_kernel_window(
            self._kernel, self._interdomain_transform, conjugate_right=True,
            scale_kernel=(1, -1), shift_kernel=(0, 0),
            scale_window=1, shift_window=0
        )

    def augmented_covariance(self):
        return augment(self.covariance(), self.pseudo_covariance())

    def augmented_interdomain_covariance(self):
        return self.interdomain_covariance().augment()


class NoiseInducingVariable(InducingVariable):
    def __init__(self, locations, interdomain_transform):
        super().__init__(locations, interdomain_transform)

    def covariance(self):
        return integrate_window(
            self._interdomain_transform, self._interdomain_transform,
            conjugate_right=True, scale_right=1, scale_left=1,
            shift_left=0, shift_right=0
        )

    def pseudo_covariance(self):
        return integrate_window(
            self._interdomain_transform, self._interdomain_transform,
            conjugate_right=False, scale_right=1, scale_left=1,
            shift_left=0, shift_right=0
        )

    def interdomain_covariance(self):
        return self._interdomain_transform.conjugate()

    def augmented_covariance(self):
        return augment(self.covariance(), self.pseudo_covariance())

    def augmented_interdomain_covariance(self):
        return self.interdomain_covariance().augment()
