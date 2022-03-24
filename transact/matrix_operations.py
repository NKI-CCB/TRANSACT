""" Matrix Operations

@author: Soufiane Mourragui

Example
-------
    
Notes
-------
BEWARE : _center_kernel is set to n_s instead of n_s - 1. Might be problematic.

References
-------

"""

import numpy as np

def _sqrt_matrix(m):
    s,u = np.linalg.eigh(m)
    # Clip at 0 to avoid negative value
    return u.dot(np.diag(np.sqrt(s.clip(0)))).dot(u.T)


def _center_kernel(k):
    n_s = k.shape[0]
    n_t = k.shape[1]
    return centering_matrix(n_s).dot(k).dot(centering_matrix(n_t))


def _right_center_kernel(k):
    n_t = k.shape[1]
    return k.dot(centering_matrix(n_t))


def _left_center_kernel(k):
    n_s = k.shape[0]
    return centering_matrix(n_s).dot(k)


def centering_matrix(n):
    """
    Computes the centering matrix of size n, i.e. the matrix with (n-1)/n on the diagonal
    and -1/n elsewhere, corresponding to a one-side centering of kernel matrix.

    -------
    n: int
        Size of the matrix to center.

    Returned Values
    -------
    C_n: np.ndarray of size (n,n)
        Centering matrix.
    """
    return np.identity(n) - (np.ones(shape=(n,n)) / n)


def _norm_matrix(m):
    column_norm = np.linalg.norm(m, axis=0)
    return m.dot(np.diag(1/column_norm))
