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


def mallow_kernel_wrapper(n_jobs=1):

    def mallow_kernel(x, y=None, gamma=None):
        print('START MALLOW COMPUTATION')
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
        ))

    return mallow_kernel