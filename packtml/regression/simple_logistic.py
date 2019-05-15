# -*- coding: utf-8 -*-

from __future__ import absolute_import

from sklearn.utils.validation import check_X_y, check_array

import numpy as np

from packtml.utils.extmath import log_likelihood, logistic_sigmoid
from packtml.utils.validation import assert_is_binary
from packtml.base import BaseSimpleEstimator

__all__ = [
    'SimpleLogisticRegression'
]

try:
    xrange
except NameError:  # py 3 doesn't have an xrange
    xrange = range


class SimpleLogisticRegression(BaseSimpleEstimator):
    """Simple logistic regression.

    This class provides a very simple example of straight forward logistic
    regression with an intercept. There are few tunable parameters aside from
    the number of iterations, & learning rate, and the model is fit upon
    class initialization.

    Parameters
    ----------
    X : array-like, shape=(n_samples, n_features)
        The array of predictor variables. This is the array we will use
        to regress on ``y``.

    y : array-like, shape=(n_samples,)
        This is the target array on which we will regress to build
        our model. It should be binary (0, 1).

    n_steps : int, optional (default=100)
        The number of iterations to perform.

    learning_rate : float, optional (default=0.001)
        The learning rate.

    loglik_interval : int, optional (default=5)
        How frequently to compute the log likelihood. This is an expensive
        operation--computing too frequently will be very expensive.

    Attributes
    ----------
    theta : array-like, shape=(n_features,)
        The coefficients

    intercept : float
        The intercept term

    log_likelihood : list
        A list of the iterations' log-likelihoods
    """
    def __init__(self, X, y, n_steps=100, learning_rate=0.001,
                 loglik_interval=5):
        X, y = check_X_y(X, y, accept_sparse=False,  # keep dense for example
                         y_numeric=True)

        # we want to make sure y is binary since that's all our example covers
        assert_is_binary(y)

        # X should be centered/scaled for logistic regression, much like
        # with linear regression
        means, stds = X.mean(axis=0), X.std(axis=0)
        X = (X - means) / stds

        # since we're going to learn an intercept, we can cheat and set the
        # intercept to be a new feature that we'll learn with everything else
        X_w_intercept = np.hstack((np.ones((X.shape[0], 1)), X))

        # initialize the coefficients as zeros
        theta = np.zeros(X_w_intercept.shape[1])

        # now for each step, we compute the inner product of X and the
        # coefficients, transform the predictions with the sigmoid function,
        # and adjust the weights by the gradient
        ll = []
        for iteration in xrange(n_steps):
            preds = logistic_sigmoid(X_w_intercept.dot(theta))
            residuals = y - preds  # The error term
            gradient = X_w_intercept.T.dot(residuals)

            # update the coefficients
            theta += learning_rate * gradient

            # you may not always want to do this, since it's expensive. Tune
            # the error_interval to increase/reduce this
            if (iteration + 1) % loglik_interval == 0:
                ll.append(log_likelihood(X_w_intercept, y, theta))

        # recall that our theta includes the intercept, so we need to pop
        # that off and store it
        self.intercept = theta[0]
        self.theta = theta[1:]
        self.log_likelihood = ll
        self.column_means = means
        self.column_std = stds

    def predict_proba(self, X):
        """Generate the probabilities that a sample belongs to class 1"""
        X = check_array(X, accept_sparse=False, copy=False)  # type: np.ndarray

        # make sure dims match
        theta = self.theta
        if theta.shape[0] != X.shape[1]:
            raise ValueError("Dim mismatch in predictors!")

        # scale the data appropriately
        X = (X - self.column_means) / self.column_std

        # creates a copy
        return logistic_sigmoid(np.dot(X, theta.T) + self.intercept)

    def predict(self, X):
        return np.round(self.predict_proba(X)).astype(int)
