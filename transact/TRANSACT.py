""" TRANSACT 

@author: Soufiane Mourragui <soufiane.mourragui@gmail.com>

Main method for the domain adaptation step.

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
import scipy
import pandas as pd
from sklearn.metrics.pairwise import kernel_metrics
from joblib import Parallel, delayed

from sklearn.model_selection import GridSearchCV, KFold
from sklearn.linear_model import Ridge, ElasticNet, Lasso, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import clone

from transact.pv_computation import PVComputation
from transact.interpolation import Interpolation
from transact.kernel_computer import KernelComputer


class TRANSACT:
    """
    Perform the whole domain adaptation step, i.e.:
        - Perform KernelPCA on source and target independently.
        - Align kernel principal components geometrically to find directions
        that are shared across source and target. This yields Principal 
        Vectors (PV).
        - Interpolate between source and target principal vectors to balance
        effect of source and target.
        - Compute the optimal interpolation time, i.e. time that minimizes the
        discrepancies in marginal between source and target.
        - Out-of-sample extension: project new dataset onto the consensus feature.

    Features to be added in future:
        - Prediction. Train classifier on source projected.
        - Vizualisation: tSNE plot of consensus feature.
        - Permutation test: how significant are the canonical angles between 
        principal vectors.
        - Ad-hoc kernel that can be passed as functional to use kernels not 
        already implemented.

    Parameters
    ----------
    kernel : str
        Name of the kernel to be used in the whole algorithm. Has to be compliant with
        scikit-learn (https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.kernel_metrics.html#sklearn.metrics.pairwise.kernel_metrics)
    
    kernel_params : dict
        Parameters of the kernel (degree for polynomial kernel, gamma for RBF). 
        Naming has to be compliant with scikit-learn.

    n_components : int or dict
        Number of components for kernel PCA. If int, then same number of components
        for source and target. If dict, then must be of the form {'source':int,
        'target':int} with indication of number of components for source and target.

    n_pv : int
        Number of principal vectors.

    method : str, default to 'two-stage'
        Method used for computing the principal vectors. Only 'two-stage' has been implemented.

    step: int, default to 100
        Number of interpolation steps.

    n_jobs: int, default to 1
        Number of concurrent threads to use for tasks that can be parallelized.

    verbose: bool, default to False (can be int)
        Degree of verbosity.
    """

    def __init__(self,
                kernel='linear',
                kernel_params=None,
                n_components=None,
                n_pv=None,
                method='two-stage',
                step=100,
                n_jobs=1,
                verbose=False):

        self.kernel = kernel
        self.kernel_params_ = kernel_params or {}
        self.kernel_values_ = KernelComputer(self.kernel, self.kernel_params_)

        self.source_data_ = None
        self.target_data_ = None

        self.is_fitted = False

        self.n_components = n_components
        self.n_pv = n_pv
        self.method = method
        self.step = step

        self.predictive_clf = None

        self.n_jobs = n_jobs
        self.verbose = verbose


    def fit(self,
            source_data,
            target_data,
            n_components=None,
            n_pv=None,
            method='two-stage',
            step=100,
            with_interpolation=True,
            left_center=True):

        """
        Fit the domain adaptive classifier, i.e.:
            - Compute Gram matrix (kernel_values_).
            - Compute principal vectors (principal_vectors_).
            - Interpolate between the PVs (interpolation_).
            - Find optimal interpolation time.

        Parameters
        ----------
        source_data : np.ndarray of shape (n_source_samples, n_features), dtype=float
            source data, matrix with samples in the rows. pandas.DataFrame would
            be supported.

        target_data : np.ndarray of shape (n_target_samples, n_features), dtype=float
            target data, matrix with samples in the rows. pandas.DataFrame would
            be supported.
            WARNING: features need to be ordered in the same way as in source_data.

        n_components: int, default to None
            Number of components. If not set in __init__, and default values, then
            maximum number of principal components for source and target will be considered.

        n_pv: int, default to None
            Number of Principal Vectors. If not set in __init__, and default values, then
            maximum number of PV will be computed.

        method : str, default to 'two-stage'
            Method used for computing the principal vectors. Only 'two-stage' has been implemented.

        step: int, default to 100
            Number of interpolation steps.

        with_interpolation: bool, default to True
            Bool indicating whether interpolation shall also be fitted. Useful for just computing PV
            prior to null distribution fitting (and choose of PV number).

        left_center: bool, default to True
            Bool indicating whether the output should be mean-centered, i.e. whether source and target
            consensus features values (or PVs if no interpolation) must have an independent mean-centering.

        Returns
        -------
        self : TRANSACT
            Fitted instance.
        """

        # Save parameters
        self.source_data_ = source_data
        self.target_data_ = target_data
        self.method = method or self.method
        self.n_components = n_components or self.n_components
        self.n_pv = n_pv or self.n_pv
        self.step = step or self.step
        self.left_center = left_center

        # Compute kernel values
        self.kernel_values_.fit(source_data, target_data, center=False)

        # Compute principal vectors
        self.principal_vectors_ = PVComputation(self.kernel, self.kernel_params_)
        self.principal_vectors_.fit(self.source_data_,
                                    self.target_data_,
                                    method=self.method,
                                    n_components=self.n_components,
                                    n_pv=self.n_pv)

        # Stop here if interpolation should not be computed.
        if not with_interpolation:
            return self

        # Set up interpolation scheme
        self.interpolation_ = Interpolation(self.kernel, self.kernel_params_)
        self.interpolation_.fit(self.principal_vectors_, self.kernel_values_)

        # Compute optimal interpolation time
        self._compute_optimal_time(step=self.step, left_center=self.left_center)

        self.is_fitted = True

        return self


    def null_distribution_pv_similarity(self, method='gene_shuffling', n_iter=100):
        """
        Generate a null distribution for the PV similarity function.
            - Gene shuffling: genes get shuffled in source to destroy any structure existing
            at the gene-level while preserving the sample structure. PV get recomputed and 
            similarity is saved.

        Parameters
        ----------
        method : string, default to gene_shuffling
            Method used for generating the null distribution.
            Only method developped: gene_shuffling

        n_iter: int, default to 100
            Number of iterations

        Returns
        -------
        np.ndarray, dtype=float, shape (n_iter, n_pv)
            Array containing the distribution of similarity after shuffling. Each row
            contains the values of one shuffling across PVs.
        """

        if method.lower() == 'gene_shuffling':
            null_method = self._gene_shuffling
        else:
            raise NotImplementedError('%s is not a proper method for generating null distribution'%(method))

        null_distribution = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)\
                                    (delayed(null_method)() for _ in range(n_iter))

        return np.array(null_distribution)


    def _gene_shuffling(self):
        perm = np.random.permutation(self.source_data_.shape[1])
        pv = PVComputation(self.kernel, self.kernel_params_)
        pv.fit(self.source_data_[:,perm],
            self.target_data_,
            method=self.method,
            n_components=self.n_components,
            n_pv=self.n_pv)

        return np.cos(pv.canonical_angles)


    def fit_predictor(self, X, y, alpha_values=None, l1_ratio=0.5):
        """
        Project X on consensus features and train a predictor for y

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features), dtype=float
            Dataset to project. Features should be ordered in same way as in source_data
            and target_data.

        y : np.ndarray of shape (n_samples, 1), dtype=float
            Output to predict

        Returns
        -------
        """
        self.alpha_values = alpha_values if alpha_values is not None else np.logspace(-10,5,34)
        self.l1_ratio_values = [0.2, 0.4, 0.5, 0.6, 0.8, 0.9, 1.]
        param_grid ={
            'regression__alpha': self.alpha_values,
            'regression__l1_ratio': self.l1_ratio_values
        }

        #Grid search setup
        self.predictive_clf = GridSearchCV(Pipeline([
                                ('scaler', StandardScaler(with_mean=True, with_std=False)),
                                ('regression', ElasticNet())
                                ]),\
                                cv=10,
                                n_jobs=self.n_jobs,
                                param_grid=param_grid,
                                verbose=self.verbose,
                                scoring='neg_mean_squared_error')
        self.predictive_clf.fit(self.transform(X, center=False), y)

        return self


    def compute_pred_performance(self, X, y, cv=10):
        """
        Compute predictive performance of predictive model by cross-validation
        on X and y.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features), dtype=float
            Dataset to project. Features should be ordered in same way as in source_data
            and target_data.

        Returns
        -------
        np.ndarray of shape (n_samples, n_pv), dtype=float
            Dataset projected on consensus features.
        """

        kf = KFold(n_splits=cv, shuffle=True)
        X_projected = self.transform(X)

        if self.predictive_clf is None:
            print('BEWARE: NOT FITTED INSTANCE')
            self.fit_predictor(X,y)
        clf = clone(self.predictive_clf)

        y_predicted = np.zeros(X.shape[0])
        for train_index, test_index in kf.split(X_projected):
            clf.fit(X_projected[train_index], y[train_index])
            y_predicted[test_index] = clf.predict(X_projected[test_index])

        return scipy.stats.pearsonr(y_predicted, y)


    def predict(self, X):
        return self.predictive_clf.predict(self.transform(X, center=False))


    def transform(self, X, center=False):
        """
        Project a dataset X onto the consensus features.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features), dtype=float
            Dataset to project. Features should be ordered in same way as in source_data
            and target_data.

        Returns
        -------
        np.ndarray of shape (n_samples, n_pv), dtype=float
            Dataset projected on consensus features.
        """
        return self.interpolation_.transform(X, self.optimal_time, center)


    def _compute_optimal_time(self, step=100, left_center=True):
        # Based on Kolmogorov Smirnov statistics, find interpolation time

        # Compute the interpolated values
        interpolated_values = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)\
                            (delayed(self.interpolation_.project_data)(s/step, center=left_center)
                                for s in range(step+1))
        interpolated_values = np.array(interpolated_values).transpose(2,0,1)
        source_interpolated_values = interpolated_values[:,:,:self.source_data_.shape[0]]
        target_interpolated_values = interpolated_values[:,:,self.source_data_.shape[0]:]

        self.optimal_time = []
        self.ks_statistics = []
        self.ks_p_values = []

        # For each PV, find the time when interpolation has the largest overlap.
        for source_pv, target_pv in zip(source_interpolated_values, target_interpolated_values):
            self.ks_statistics.append([])
            for s, t in zip(source_pv, target_pv):
                self.ks_statistics[-1].append(scipy.stats.ks_2samp(s,t))
            self.ks_statistics[-1] = list(zip(*self.ks_statistics[-1]))
            self.ks_p_values.append(self.ks_statistics[-1][-1])
            self.ks_statistics[-1] = self.ks_statistics[-1][0]
            self.optimal_time.append(np.argmin(self.ks_statistics[-1])/step)

        # Save the different statistics
        self.optimal_time = np.array(self.optimal_time) # Optimal tau for each PV.
        self.ks_statistics = np.array(self.ks_statistics) # Computed KS statistics between each PV.
        self.ks_p_values = np.array(self.ks_p_values) # Corresponding p_values.