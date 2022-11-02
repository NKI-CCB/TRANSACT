""" Kernel Computer

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

from transact.matrix_operations import _sqrt_matrix, _center_kernel, _right_center_kernel
from transact.alternative_kernels import mallow_kernel_wrapper, kendall_kernel_wrapper, random_spearman_kernel, spearman_kernel


class KernelComputer():

    def __init__(self, kernel, kernel_params={}, n_jobs=1):

        self.kernel_ = kernel
        self.n_jobs = n_jobs
        if self.kernel_.lower() == 'mallow':
            self.kernel_ = mallow_kernel_wrapper(self.n_jobs)
        elif self.kernel_.lower() == 'kendall':
            self.kernel_ = kendall_kernel_wrapper(self.n_jobs)
        elif self.kernel_.lower() == 'spearman':
            self.kernel_ = spearman_kernel
        elif self.kernel_.lower() == 'random_spearman':
            self.kernel_ = random_spearman_kernel
        else:
            self.kernel_ = kernel_metrics()[kernel]
        self.kernel_params_ = kernel_params
        self._source_data = None
        self._target_data = None
        self._data = {}
        self.center=False

        self._empty_kernel_values()


    def fit(self, X_source, X_target, center=True):

        self._source_data = X_source
        self._target_data = X_target
        self._data = {'source': X_source, 'target': X_target}
        self.center = center

        self._compute_kernel(center=self.center)

        return self


    def transform(self, X, center=False, right_center=False):
        """
        Returns kernel matrix with X in rows and (source, target) in columns
        """
        K_with_source = self.kernel_(X, self._source_data, **self.kernel_params_)
        K_with_target = self.kernel_(X, self._target_data, **self.kernel_params_)

        if center:
            K_with_source = _center_kernel(K_with_source)
            K_with_target = _center_kernel(K_with_target)
        elif right_center:
            K_with_source = _right_center_kernel(K_with_source)
            K_with_target = _right_center_kernel(K_with_target)
        
        return np.concatenate([K_with_source, K_with_target], axis=1)


    def _compute_kernel(self, center=False):
        # Global kernel matrix
        self.kernel_matrix_ = self.kernel_(np.concatenate([self._source_data, self._target_data]),
                                            **self.kernel_params_)
        
        # Individual kernel matrices for source, target, and cross-over.
        n_source_samples = self._source_data.shape[0]
        self.k_s = self.kernel_matrix_[:n_source_samples,:n_source_samples]
        self.k_t = self.kernel_matrix_[n_source_samples:,n_source_samples:]
        self.k_st = self.kernel_matrix_[:n_source_samples,n_source_samples:]

        if center:
            self.k_s = _center_kernel(self.k_s)
            self.k_t = _center_kernel(self.k_t)
            self.k_st = _center_kernel(self.k_st)
            
        self.k_ts = self.k_st.T

        self.kernel_submatrices = {
            'source': self.k_s,
            'target': self.k_t,
            'source-target': self.k_st,
            'target-source': self.k_ts
        }


    def _empty_kernel_values(self):

        self.kernel_matrix_ = None
        self.kernel_submatrices = None
        self.k_s = None
        self.k_t = None
        self.k_st = None
        self.k_ts = None


    @property
    def source_data(self):
        return self._source_data

    @source_data.setter
    def source_data(self, X):
        self._source_data = X
        self._data['source'] = X

        if X is None:
            self._empty_kernel_values()
        elif self._target_data is not None:
            self._compute_kernel(center=self.center)

    @property
    def target_data(self):
        return self._target_data

    @target_data.setter
    def target_data(self, X):
        self._target_data = X
        self._data['target'] = X

        if X is None:
            self._empty_kernel_values()
        elif self._source_data is not None:
            self._compute_kernel(center=self.center)

    @property
    def data(self):
        return self._data
    
