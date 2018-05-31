# -*- coding: utf-8 -*-

from __future__ import absolute_import

import numpy as np

__all__ = [
    'log_likelihood',
    'logistic_sigmoid'
]


def log_likelihood(X, y, w):
    """Compute the log-likelihood function.

    Computes the log-likelihood function over the training data.
    The key to the log-likelihood is that the log of the product of
    likelihoods becomes the sum of logs. That is (in pseudo-code),

        np.log(np.product([f(i) for i in range(N)]))

    is equivalent to:

        np.sum([np.log(f(i)) for i in range(N)])

    The log-likelihood function is used in computing the gradient for
    our loss function since the derivative of the sum (of logs) is equivalent
    to the sum of derivatives, which simplifies all of our math.

    Parameters
    ----------
    X : np.ndarray, shape=(n_samples, n_features)
        The training data.

    y : np.ndarray, shape=(n_samples,)
        The target vector of 1s or 0s.

    w : np.ndarray, shape=(n_features,)
        The vector of feature weights (coefficients)

    References
    ----------
    .. [1] For a very thorough explanation of the log-likelihood function, see
           https://www.coursera.org/learn/ml-classification/lecture/1ZeTC/very-optional-expressing-the-log-likelihood
    """
    weighted = X.dot(w)
    return (y * weighted - np.log(1. + np.exp(weighted))).sum()


def logistic_sigmoid(x):
    """The logistic function.

    Compute the logistic (sigmoid) function over a vector, ``x``.

    Parameters
    ----------
    x : np.ndarray, shape=(n_samples,)
        A vector to transform.
    """
    return 1. / (1. + np.exp(-x))
