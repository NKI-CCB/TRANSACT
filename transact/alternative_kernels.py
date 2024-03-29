""" Alternative Kernels

@author: Soufiane Mourragui

Implementation of kernels that are not native to scikit-learn.

Example
-------
    
Notes
-------

References
-------

"""


import numpy as np
import scipy
import scipy.stats
from joblib import Parallel, delayed
import tqdm


def compute_number_discordant(x,y):
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    size = x.size
    perm = np.argsort(y)  # sort on y and convert y to dense ranks
    x, y = x[perm], y[perm]
    y = np.r_[True, y[1:] != y[:-1]].cumsum(dtype=np.intp)

    # stable sort on x and convert x to dense ranks
    perm = np.argsort(x, kind='mergesort')
    x, y = x[perm], y[perm]
    x = np.r_[True, x[1:] != x[:-1]].cumsum(dtype=np.intp)

    return scipy.stats._stats._kendall_dis(x, y)  # discordant pairs


def compute_number_discordant_row(gamma,X,y):
    return [
        np.exp(- gamma * compute_number_discordant(e,y))
        for e in X
    ]


def kendall_kernel_wrapper(n_jobs=1):

    def row_wise_kernel(x,y):
        return [scipy.stats.kendalltau(x,y_val)[0] for y_val in y]

    def kendall_kernel(X, y=None):
        if y is None:
            y = X

        return np.array(Parallel(n_jobs=n_jobs, verbose=1)(
            delayed(row_wise_kernel)(x,y) for x in X
        ))

    return kendall_kernel

def mallow_kernel_wrapper(n_jobs=1):

    def mallow_kernel(x, y=None, gamma=None):
        x = np.asarray(x)
        x = x.reshape(1 if len(x.shape) == 1 else x.shape[0],-1)
        if y is None:
            y = x
        else:
            y = np.asarray(y)
            y = y.reshape(1 if len(y.shape) == 1 else y.shape[0],-1)
        
        return np.array(Parallel(n_jobs=n_jobs, verbose=1)(
            delayed(compute_number_discordant_row)(gamma, x, f)
            for f in y
        )).T

    return mallow_kernel

def spearman_kernel(X, y=None):
    """
    Linear kernel between ranks of X and y.
    """
    p = X.shape[1]

    if len(X.shape) == 1:
        X = X.reshape(1,-1)
    X_ranks = scipy.stats.rankdata(X, axis=1, method='ordinal').astype(np.float64) - (p+1)/2
    
    if y is None:
        y_ranks = X_ranks
    else:
        if len(y.shape) == 1:
            y = y.shape(-1,1)
        y_ranks = scipy.stats.rankdata(y, axis=1, method='ordinal').astype(np.float64) - (p+1)/2
        
    return X_ranks.dot(y_ranks.T) / p


def single_loop_random_spearman_kernel(X,y=None, n_iter=250, left_random=True, right_random=True, n_jobs=1):
    y_provided = y is not None
    if y is None:
        y = X

    if left_random:
        min_diff_X = np.unique(X)
        min_diff_X = np.min(min_diff_X[1:] - min_diff_X[:-1])
        sigma_X = min_diff_X / 10.
    else:
        sigma_X = 0.

    if right_random:
        min_diff_Y = np.unique(y)
        min_diff_Y = np.min(min_diff_Y[1:] - min_diff_Y[:-1])
        sigma_Y = min_diff_Y / 10.
    else:
        sigma_Y = 0.

    X_rand = X + np.random.normal(0,sigma_X, X.shape)
    y_rand = y + np.random.normal(0,sigma_Y, y.shape) if y_provided else X_rand
    kernel_matrix = spearman_kernel(X_rand, y_rand)
    
    for _ in tqdm.tqdm(range(n_iter-1), position=0, leave=True):
        X_rand = X + np.random.normal(0,sigma_X, X.shape)
        y_rand = y + np.random.normal(0,sigma_Y, y.shape) if y_provided else X_rand
        kernel_matrix = kernel_matrix + spearman_kernel(X_rand, y_rand)

    return kernel_matrix


def random_spearman_kernel(X,y=None, n_iter=250, left_random=True, right_random=True, n_jobs=1):
    """
    Return expectation of Spearman correlation when Gaussian model is added
    to X and y. Standard deviation of Gaussian model is based on 10% of the
    minimum difference observed in X.

    Parameters
    ----------
    X : np.ndarray
        Array to study, with samples in the rows.
    y: np.ndarray
        Second array to compute kernel agains, default to None. If one wants
        to compute the kernel of X with itself, please set y=None (speeds up
        computation).
    n_iter: int, default to 50
        Number of Monte-Carlo samplings.
    left_random: bool, default to True
        Whether random noise must be added to X.
    right_random: bool, default to True
        Whether random noise must be added to Y.

    Returns
    -------
    np.ndarray with kernel values
    """
    
    iter_breakdown = np.linspace(0,n_iter,n_jobs+1).astype(int)
    iter_breakdown = iter_breakdown[1:] - iter_breakdown[:-1]
    kernel_matrix = np.sum(
        Parallel(n_jobs=n_jobs, verbose=10, require='sharedmem')(
            delayed(single_loop_random_spearman_kernel)(X,y, it, left_random, right_random, n_jobs)
            for it in iter_breakdown
            ), 
        axis=0
    )

    return kernel_matrix / n_iter

def random_spearman_kernel_wrapper(n_iter):

    def kernel(X, y=None):
        return random_spearman_kernel(X,y,n_iter)

    return kernel
