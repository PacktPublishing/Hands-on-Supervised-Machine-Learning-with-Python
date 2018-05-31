# -*- coding: utf-8 -*-

from __future__ import absolute_import

from numpy import linalg as la

__all__ = [
    'l2_norm'
]


def l2_norm(X, axis=0):
    """Compute the L2 (Euclidean) norm of a matrix.

    Computes the L2 norm along the specified axis. If axis is 0,
    computes the norms along the columns. If 1, computes along the
    rows.

    Parameters
    ----------
    X : array-like, shape=(n_samples, n_features)
        The matrix on which to compute the norm.

    axis : int, optional (default=0)
        The axis along which to compute the norm. 0 is for columns,
        1 is for rows.
    """
    return la.norm(X, ord=None, axis=axis)
