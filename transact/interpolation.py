""" Interpolation

@author: Soufiane Mourragui

Example
-------
    
Notes
-------

References
-------

"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import kernel_metrics
from sklearn.decomposition import KernelPCA
from sklearn.metrics.pairwise import pairwise_kernels, kernel_metrics

from transact.matrix_operations import _sqrt_matrix, _center_kernel, centering_matrix
from transact.pv_computation import PVComputation
from transact.kernel_computer import KernelComputer



class Interpolation:

    def __init__(self, kernel, kernel_params={}, n_jobs=1):

        self.kernel = kernel
        self.kernel_params_ = kernel_params

        self._principal_vectors = None
        self.kernel_values_ = None

    def fit(self,
            principal_vectors,
            kernel_values,
            n_pv=None):

        # Feed values to the interpolation scheme
        self._principal_vectors = principal_vectors
        self.kernel_values_ = kernel_values

        self.n_pv = n_pv or (self._principal_vectors.n_pv or min(self._principal_vectors.n_components.values()))

        self._compute_coef_matrix()

        return self

    def project_data(self, tau, source_only=False, target_only=False, center=True):
        """
        Compute the source and/or target data projected on the interpolated path at time
        tau.

        Parameters
        -------

        Returned Values
        -------

        """

        if type(tau) == int or type(tau) == float:
            tau = [tau]*self.n_pv
        tau = np.array(tau)

        sample_coef_matrix = self.pv_matrix_
        if center:
            sample_coef_matrix = self.centering_matrix_.dot(sample_coef_matrix)

        if source_only:
            return sample_coef_matrix[:self.n_source_samples].dot(self._angular_interpolation_matrix(tau))
        elif target_only:
            return sample_coef_matrix[self.n_source_samples:].dot(self._angular_interpolation_matrix(tau))
        else:
            return sample_coef_matrix.dot(self._angular_interpolation_matrix(tau))


    def transform(self, X, tau, center=True):
        
        if type(tau) == int or type(tau) == float:
            tau = [tau]*self.n_pv
        tau = np.array(tau)
        
        K = self.kernel_values_.transform(X, center=center, right_center=True)
        return K.dot(self.coef_matrix_).dot(self._angular_interpolation_matrix(tau))


    def _angular_interpolation_matrix(self, tau):
        assert tau.shape == (self.n_pv,)

        return np.block([[np.diag(self._gamma_interpolations(tau))],
                        [np.diag(self._xi_interpolations(tau))]])


    def _gamma_interpolations(self, tau):
        num = np.sin((1-tau)*self._principal_vectors.canonical_angles)
        den = np.sin(self._principal_vectors.canonical_angles)
        return num / den


    def _xi_interpolations(self, tau):
        num = np.sin(tau*(self._principal_vectors.canonical_angles))
        den = np.sin(self._principal_vectors.canonical_angles)
        return num / den


    def _compute_coef_matrix(self):
        """
        Compute different matrices of used later:
        - kernel_matrix_: kernel values between source and target data (stacked).
        - centering_matrix_: global centering matrix.
        - coef_matrix_: coefficients for source and target principal components
        - pv_matrix_: source and target principal vectors coefficients later used for interpolation.
        """
        gamma_source = self._principal_vectors.gamma_coef['source']
        gamma_target = self._principal_vectors.gamma_coef['target']
        self.n_source_samples = gamma_source.shape[1]
        self.n_target_samples = gamma_target.shape[1]

        self.kernel_matrix_ = np.block([
            [
                self.kernel_values_.k_s,
                self.kernel_values_.k_st
            ],
            [
                self.kernel_values_.k_ts,
                self.kernel_values_.k_t
            ]
        ])

        self.centering_matrix_ = np.block([
            [
                centering_matrix(self.n_source_samples),
                np.zeros((self.n_source_samples, self.n_target_samples))
            ],
            [
                np.zeros((self.n_target_samples, self.n_source_samples)),
                centering_matrix(self.n_target_samples)
            ]
        ])

        self.coef_matrix_ =  np.block([
            [
                gamma_source.T[:,:self.n_pv],
                np.zeros((self.n_source_samples, self.n_pv))
            ],
            [
                np.zeros((self.n_target_samples, self.n_pv)),
                gamma_target.T[:,:self.n_pv]
            ]
        ])

        self.pv_matrix_ = self.kernel_matrix_.dot(self.centering_matrix_).dot(self.coef_matrix_)

    @property
    def principal_vectors(self):
        return self._principal_vectors

    @principal_vectors.setter
    def principal_vectors(self, pv):
        self._principal_vectors = pv

        if not self.is_pv_fitted_:
            self._compute_coef_matrix
        self.is_pv_fitted_ = True
