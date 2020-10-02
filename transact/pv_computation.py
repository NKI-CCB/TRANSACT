""" PVComputation

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
from sklearn.metrics.pairwise import kernel_metrics
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge, ElasticNet

from transact.matrix_operations import _sqrt_matrix, _center_kernel, _right_center_kernel, _left_center_kernel
from transact.kernel_computer import KernelComputer


class PVComputation:
    """

    Attributes
    -------
    """

    def __init__(self, kernel, kernel_params={}, n_components=None, n_pv=None):

        self.gamma_coef = None
        self.alpha_coef = None
        self.beta_coef = None
        self.canonical_angles = None

        self.kernel = kernel
        self.kernel_ = kernel_metrics()[kernel]
        self.kernel_params_ = kernel_params

        self.kernel_values_ = KernelComputer(self.kernel, self.kernel_params_)

        # Put n_components in dictionary format.
        self.n_components = n_components
        if type(self.n_components) == int:
            self.n_components = {
                s:self.n_components for s in ['source', 'target']
            }
        self.n_pv = n_pv

        self.n_jobs = 1

    def fit(self, source_data, target_data, method='two-stage', n_components=None, n_pv=None):
        """
        Computes the kernel principal vectors between source and target data.

        Parameters
        -------
        source_data: numpy.ndarray, shape (n_samples, n_genes)
            Source data

        target_data: numpy.ndarray, shape (n_samples, n_genes)
            Source data

        method: str, default to "two-stage"
            Method used for computed the kernel PVs, either "two-stage" (first kernel PCA, then
            alignment), or "direct" (direct minimization)

        n_components: int, default to None
            Number of components taken into the decomposition.

        Returned Values
        -------
        self: returns an instance of self.
        """

        # Compute kernel matrices
        self.kernel_values_.fit(source_data, target_data, center=True)

        if method == 'two-stage':
            self._two_stage_computation(n_components, n_pv)
        elif method == 'direct':
            self._direct_computation(n_components)

        return self
        

    def transform(self, X, right_center=False):
        """
        Project data X on source and target kernel principal vectors

        Parameters
        -------
        X: numpy.ndarray, shape (n_samples, n_genes)
            Data to project

        right_center: Boolean, default to False
            Whether data should be implicitly mean centered

        Returned Values
        -------
        Dictionary with 'source' and 'target' as keys, and projected arrays as values.
        """

        X_projected = {}
        for t in ['source', 'target']:
            X_projected[t] = self._project_PV_from_data(X, t, right_center)

        return X_projected


    def fit_transform(self, source_data, target_data, method='two-stage', n_components=None, n_pv=None):
        """
        Computes the kernel principal vectors between source and target data.

        -------
        source_data: numpy.ndarray, shape (n_samples, n_genes)
            Source data

        target_data: numpy.ndarray, shape (n_samples, n_genes)
            Source data

        method: str, default to "two-stage"
            Method used for computed the kernel PVs, either "two-stage" (first kernel PCA, then
            alignment), or "direct" (direct minimization)

        n_components: int or dictionary, default to None
            Number of components taken into account for PCA. Can be int (if same number of components
            for source or target) or dictionary with {'source': int, 'target':int} indicating the
            number of source and target principal components.

        Returned Values
        -------
        source_projected: dictionary

        target_projected: dictionary
        """

        self.fit(source_data, target_data, method, n_components)

        source_projected = {
            'source': self._project_PV_from_data(source_data, 'source'),
            'target': self._project_PV_from_data(source_data, 'target')
        }
        
        target_projected = {
            'source': self._project_PV_from_data(target_data, 'source'),
            'target': self._project_PV_from_data(target_data, 'target')
        }

        return source_projected, target_projected


    def _two_stage_computation(self, n_components=None, n_pv=None):

        self.n_components = n_components or self.n_components
        if self.n_components is None or type(self.n_components) == int:
            self.n_components = {
                s:self.n_components for s in ['source', 'target']
            }

        self.n_pv = n_pv or (self.n_pv or min(self.n_components.values()))

        ## First step: Kernel PCA
        self._dim_reduction()
        
        ## Second step: Align based on cosine similarity
        self._align_principal_components()


    def _dim_reduction(self):
        self.dim_reduc_clf_ = {}
        self.alpha_coef = {}

        # Independent processing of source and target
        for t in ['source', 'target']:
            # Reduce dimensionality using kernelPCA.
            self.dim_reduc_clf_[t] = KernelPCA(self.n_components[t],
                                            kernel=self.kernel,
                                            n_jobs=self.n_jobs,
                                            **self.kernel_params_)
            self.dim_reduc_clf_[t].fit(self.kernel_values_.data[t])

            # Save kernel PCA coefficients
            self.alpha_coef[t] = self.dim_reduc_clf_[t].alphas_ / np.sqrt(self.dim_reduc_clf_[t].lambdas_)


    def _align_principal_components(self):
        self.cosine_similarity_ = self.alpha_coef['source'].T.dot(self.kernel_values_.k_st).dot(self.alpha_coef['target'])
        
        beta_s, theta, beta_t = np.linalg.svd(self.cosine_similarity_)
        self.beta_coef = {}
        self.beta_coef['source'] = beta_s
        self.beta_coef['target'] = beta_t.T # Due to definition of SVD by matplotlib

        # Computation of gamma coefficients
        self.gamma_coef = {}
        for t in ['source', 'target']:
            self.gamma_coef[t] = self.beta_coef[t].T.dot(self.alpha_coef[t].T)
            self.gamma_coef[t] = self.gamma_coef[t][:self.n_pv]

        # Canonical angles
        self.canonical_angles = np.arccos(theta[:self.n_pv])


    def _direct_computation(self, n_components=None):
        print('BEWARE')
        print('DIRECT COMPUTATION HAS NOT BEEN CORRECTED')
        print('AND COME WITH SUBTANTIAL COMPUTATION PROBLEM')
        print('DO NOT USE AS SUCH')
        self.n_components = n_components or min(self.kernel_values_.source_data.shape[0], self.kernel_values_.target_data.shape[0])

        # Compute kernel cosine similarity
        self.kernel_cosine_similarity_ = np.linalg.pinv(_sqrt_matrix(self.kernel_values_.k_s)).dot(self.kernel_values_.k_st).dot(np.linalg.pinv(_sqrt_matrix(self.kernel_values_.k_t)))

        sigma_s, theta, sigma_t = np.linalg.svd(self.kernel_cosine_similarity_)
        sigma_t = sigma_t.T
        self.gamma_coef = {}
        self.gamma_coef['source'] = np.linalg.inv(_sqrt_matrix(self.kernel_values_.k_s)).dot(sigma_s[:,:self.n_components]).T
        self.gamma_coef['target'] = np.linalg.inv(_sqrt_matrix(self.kernel_values_.k_t)).dot(sigma_t[:,:self.n_components]).T

        # Canonical angles
        self.canonical_angles = theta

    def _project_PV_from_data(self, X, t, right_center=False):
        """
        Project data X on source and target kernel principal vectors

        -------
        X: numpy.ndarray, shape (n_samples, n_genes)
            Data to project

        t: str
            Type, either 'source' or 'target'

        right_center: Boolean, default to False
            Whether data should be implicitly mean centered

        Returned Values
        -------
        Dictionary with 'source' and 'target' as keys, and projected arrays as values.
        Projected arrays are of size (n_samples, n_pv)
        """
        
        K = self.kernel_(self.kernel_values_.data[t], X, **self.kernel_params_)
        K = _left_center_kernel(K)
        if right_center:
            K = _right_center_kernel(K)

        return self._project_PV_from_kernel(K,t)

    def _project_PV_from_kernel(self, K, t):
        """
        Project kernel X on source and target kernel principal vectors

        -------
        K: numpy.ndarray, shape (n_samples, n_samples)
            Kernel matrix between data from type t and specific dataset.
            Source (or target) samples in the rows (same order as given to the algorithm)
            New dataset samples in the columns 

        t: str
            Type, either 'source' or 'target'

        Returned Values
        -------
        Dictionary with 'source' and 'target' as keys, and projected arrays as values.
        Projected arrays are of size (n_samples, n_pv)
        """

        return self.gamma_coef[t].dot(K).T 