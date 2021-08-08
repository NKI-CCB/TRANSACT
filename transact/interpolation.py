""" <h3>Interpolation</h3>
Interpolation routine, once the PVs have been computed, i.e.:
<ul>
    <li> Project data on discretised geodesic flow.
    <li> Compare distributions at different projection steps.
    <li> Find optimal interpolation points and handle final projection.
</ul>

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

    def __init__(self, kernel, kernel_params={}):
        """
        Parameters
        ----------
        kernel : str, default to 'linear'
            Name of the kernel to be used in the algorithm. Has to be compliant with
            <a href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.kernel_metrics.html#sklearn.metrics.pairwise.kernel_metrics">
            scikit-learn kernel</a>, e.g., "rbf", "polynomial", "laplacian", "linear", ...

        kernel_params : dict, default to None
            Parameters of the kernel (degree for polynomial kernel, gamma for RBF).
            Naming has to be compliant with scikit-learn, e.g., {"gamma": 0.0005}.
        """

        self.kernel = kernel
        self.kernel_ = kernel_metrics()[kernel]
        self.kernel_params_ = kernel_params

        self._principal_vectors = None
        self.kernel_values_ = KernelComputer(self.kernel, self.kernel_params_)


    def fit(self,
            principal_vectors,
            kernel_values,
            n_pv=None):
        """
        Given a computed set of Principal Vectors, compute the Consensus Features (CFs).
        <br/> Specifically:
         <ul>
            <li> Project data on discretised geodesic flow.
            <li> Compare distributions at different projection steps.
            <li> Find optimal interpolation points and handle final projection.
        </ul>

        Parameters
        ----------
        principal_vectors : PVComputation
            Fitted principal vectors, computed on source and target data.

        kernel_values : KernelComputer
            Similarity matrices computed between source and target (and within source and target).

        n_pv: int, default to None
            Number of Principal Vectors. If not set here or in __init__, then maximum number of PV will be computed.

        Returns
        -------
        self : Interpolation
            Fitted instance.
        """

        # Feed values to the interpolation scheme
        self._principal_vectors = principal_vectors
        self.kernel_values_ = kernel_values

        self.n_pv = n_pv or (self._principal_vectors.n_pv or min(self._principal_vectors.n_components.values()))

        self._compute_coef_matrix()

        return self

    def project_data(self, tau, source_only=False, target_only=False, center=True):
        """
        Project the data used to fit the instance.

        Parameters
        ----------
        tau : float, np.ndarray (dtype:float) or list
            Optimal interpolation time. If list or np.ndarray, then must be of size n_pv and contains the optimal interpolation
            time for each of the PV pairs. If float, then same interpolation time used for all PV pairs.
            <br/>
            <b>WARNING:</b> In any case, the interpolation time should be between 0 and 1. 0 means projection on the source PV, 1
            means projection on the target PV.

        source_only : bool, default to False
            Whether only source data should be projected.

        target_only : bool, default to False
            Whether only target data should be projected.
            <br/>
            <b>WARNING:</b> If source_only is True, then target_only will be overlooked.

        center: bool, default to True
            Whether the data should be centered prior to projection (kernel centering).

        Returns
        -------
        np.ndarray: projection of the data, samples ordered as input in fit, with source samples coming first.
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
        """
        Project data on interpolated points.

        Parameters
        ----------
        X : np.ndarray, dtype: float
            Data to project, of dimension (n_samples, n_genes). Genes (or features) should have same ordering than used in the 
            fitting procedure.

        tau : float, np.ndarray (dtype:float) or list
            Optimal interpolation time. If list or np.ndarray, then must be of size n_pv and contains the optimal interpolation
            time for each of the PV pairs. If float, then same interpolation time used for all PV pairs.
            <br/>
            <b>WARNING:</b> In any case, the interpolation time should be between 0 and 1. 0 means projection on the source PV, 1
            means projection on the target PV.

        center: bool, default to True
            Whether the data should be centered prior to projection (kernel centering).

        Returns
        -------
        np.ndarray: projection of the data, samples ordered as input in fit, with source samples coming first.
        """
        
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
