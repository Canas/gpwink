# -*- coding: utf-8 -*-
"""
gpwink.inducing
~~~~~~~~~~~~~~~
This module create the inter-domain inducing variable approximations.
There is always only two inducing variables in the process:

- Filter inducing variable u
- Noise inducing variable v

Both approximate the filter process h and the noise process x, respectively.
"""
from gpwink_tf.integral import (integrate_kernel_window, integrate_window,
                                integrate_window_kernel_window)
from gpwink_tf.utils import augment


class InducingVariable:
    """Base class for Inducing Variable models."""

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
    """Filter Inducing Variable model"""

    def __init__(self, locations, interdomain_transform, kernel):
        """Define the Filter Inducing Variable components.
        
        :param locations: array with locations in the filter space
        :param interdomain_transform: Integrable window object
        :param kernel: Integrable kernel object
        :return: None
        """
        super().__init__(locations, interdomain_transform)
        self._kernel = kernel

    def covariance(self):
        """Covariance of the filter model. """
        if self._cov is None:
            self._cov = integrate_window_kernel_window(
                self._interdomain_transform, self._kernel,
                self._interdomain_transform, conjugate_right=True,
                scale_left=-1, scale_mid=(1, 1), scale_right=-1,
                shift_left=0, shift_mid=(0, 0), shift_right=0
            )
        return self._cov

    def pseudo_covariance(self):
        """Pseudo-covariance of the filter model. """
        if self._pcov is None:
            self._pcov = integrate_window_kernel_window(
                self._interdomain_transform, self._kernel,
                self._interdomain_transform, conjugate_right=False,
                scale_left=-1, scale_mid=(1, 1), scale_right=-1,
                shift_left=0, shift_mid=(0, 0), shift_right=0
            )
        return self._pcov

    def interdomain_covariance(self):
        """Inter-domain covariance of the filter model. """
        if self._interdomain_cov is None:
            self._interdomain_cov = integrate_kernel_window(
                self._kernel, self._interdomain_transform, conjugate_right=True,
                scale_kernel=(1, -1), shift_kernel=(0, 0),
                scale_window=1, shift_window=0
            )
        return self._interdomain_cov

    def augmented_covariance(self):
        """Augmented covariance of the filter model. """
        if self._a_cov is None:
            self._a_cov = augment(self.covariance(), self.pseudo_covariance())
        return self._a_cov

    def augmented_interdomain_covariance(self):
        """Augmented inter-domain covariance of the filter model. """
        if self._a_interdomain_cov is None:
            self._a_interdomain_cov = self.interdomain_covariance().augment()
        return self._a_interdomain_cov


class NoiseInducingVariable(InducingVariable):
    """Noise Inducing Variable model"""

    def __init__(self, locations, interdomain_transform):
        """Define the Noise Inducing Variable components.
        
        :param locations: array with locations in the noise space
        :param interdomain_transform: Integrable window object
        :return: None
        """
        super().__init__(locations, interdomain_transform)

    def covariance(self):
        """Covariance of the noise model. """
        if self._cov is None:
            self._cov = integrate_window(
                self._interdomain_transform, self._interdomain_transform,
                conjugate_right=True, scale_right=1, scale_left=1,
                shift_left=0, shift_right=0
            )
        return self._cov

    def pseudo_covariance(self):
        """Pseudo-covariance of the noise model. """
        if self._pcov is None:
            self._pcov = integrate_window(
                self._interdomain_transform, self._interdomain_transform,
                conjugate_right=False, scale_right=1, scale_left=1,
                shift_left=0, shift_right=0
            )
        return self._pcov

    def interdomain_covariance(self):
        """Inter-domain covariance of the noise model. """
        if self._interdomain_cov is None:
            self._interdomain_cov = self._interdomain_transform.conjugate()
        return self._interdomain_cov

    def augmented_covariance(self):
        """Augmented covariance of the noise model. """
        if self._a_cov is None:
            self._a_cov = augment(self.covariance(), self.pseudo_covariance())
        return self._a_cov

    def augmented_interdomain_covariance(self):
        """Augmented inter-domain covariance of the noise model. """
        if self._a_interdomain_cov is None:
            self._a_interdomain_cov = self.interdomain_covariance().augment()
        return self._a_interdomain_cov
