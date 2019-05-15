# -*- coding: utf-8 -*-

from __future__ import absolute_import

from sklearn.utils.validation import check_X_y, check_array

import numpy as np
from numpy.linalg import lstsq

from packtml.base import BaseSimpleEstimator


__all__ = [
    'SimpleLinearRegression'
]


class SimpleLinearRegression(BaseSimpleEstimator):
    """Simple linear regression.

    This class provides a very simple example of straight forward OLS
    regression with an intercept. There are no tunable parameters, and
    the model fit happens directly on class instantiation.

    Parameters
    ----------
    X : array-like, shape=(n_samples, n_features)
        The array of predictor variables. This is the array we will use
        to regress on ``y``.

    y : array-like, shape=(n_samples,)
        This is the target array on which we will regress to build
        our model.

    Attributes
    ----------
    theta : array-like, shape=(n_features,)
        The least-squares solution (the coefficients)

    rank : int
        The rank of the predictor matrix, ``X``

    singular_values : array-like, shape=(n_features,)
        The singular values of ``X``

    X_means : array-like, shape=(n_features,)
        The column means of the predictor matrix, ``X``

    y_mean : float
        The mean of the target variable, ``y``

    intercept : float
        The intercept term
    """
    def __init__(self, X, y):
        # First check X, y and make sure they are of equal length, no NaNs
        # and that they are numeric
        X, y = check_X_y(X, y, y_numeric=True,
                         accept_sparse=False)  # keep it simple

        # Next, we want to scale all of our features so X is centered
        # We will do the same with our target variable, y
        X_means = np.average(X, axis=0)
        y_mean = y.mean(axis=0)

        # don't do in place, so we get a copy
        X = X - X_means
        y = y - y_mean

        # Let's compute the least squares on X wrt y
        # Least squares solves the equation `a x = b` by computing a
        # vector `x` that minimizes the Euclidean 2-norm `|| b - a x ||^2`.
        theta, _, rank, singular_values = lstsq(X, y, rcond=None)

        # finally, we compute the intercept values as the mean of the target
        # variable MINUS the inner product of the X_means and the coefficients
        intercept = y_mean - np.dot(X_means, theta.T)

        # ... and set everything as an instance attribute
        self.theta = theta
        self.rank = rank
        self.singular_values = singular_values

        # we have to retain some of the statistics around the data too
        self.X_means = X_means
        self.y_mean = y_mean
        self.intercept = intercept

    def predict(self, X):
        """Compute new predictions for X"""
        # copy, make sure numeric, etc...
        X = check_array(X, accept_sparse=False, copy=False)  # type: np.ndarray

        # make sure dims match
        theta = self.theta
        if theta.shape[0] != X.shape[1]:
            raise ValueError("Dim mismatch in predictors!")

        # creates a copy
        return np.dot(X, theta.T) + self.intercept
