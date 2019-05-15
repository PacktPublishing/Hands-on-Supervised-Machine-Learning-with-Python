# -*- coding: utf-8 -*-

from __future__ import absolute_import

from sklearn.utils.validation import check_random_state, check_array

from numpy.linalg import solve
import numpy as np

from packtml.recommendation.base import RecommenderMixin
from packtml.base import BaseSimpleEstimator

__all__ = [
    'ALS'
]

try:
    xrange
except NameError:  # py3 does not have xrange
    xrange = range


def mse(R, X, Y, W):
    """Compute the reconstruction MSE. This is our loss function"""
    return ((W * (R - X.dot(Y))) ** 2).sum()


class ALS(BaseSimpleEstimator, RecommenderMixin):
    r"""Alternating Least Squares for explicit ratings matrices.

    Computes the ALS user factors and item factors for explicit ratings
    systems. This solves:

        R' = XY

    where ``X`` is an (m x f) matrix of user factors, and ``Y`` is an
    (f x n) matrix of item factors. Note that for very large ratings matrices,
    this can quickly grow outside the scope of what will fit into memory!

    Parameters
    ----------
    R : array-like, shape=(n_users, n_items)
        The ratings matrix. This must be an explicit ratings matrix where
        0 indicates an item that a user has not yet rated.

    factors : int or float, optional (default=0.25)
        The number of factors to learn. Default is ``0.25 * n_items``.

    n_iter : int, optional (default=10)
        The number of iterations to perform. The larger the number, the
        smaller the train error, but the more likely to overfit.

    lam : float, optional (default=0.001)
        The L2 regularization parameter. The higher ``lam``, the more
        regularization is performed, and the more robust the solution. However,
        extra iterations are typically required.

    random_state : int, None or RandomState, optional (default=None)
        The random state for seeding the initial item factors matrix, ``Y``.

    Attributes
    ----------
    X : np.ndarray, shape=(n_users, factors)
        The user factors

    Y : np.ndarray, shape=(factors, n_items)
        The item factors

    train_err : list
        The list of training MSE for each iteration performed

    lam : float
        The lambda (regularization) value.

    Notes
    -----
    If you plan to use a very large matrix, consider using a sparse CSR matrix
    to preserve memory, but you'll have to amend the ``recommend_for_user``
    function, which expects dense output.
    """
    def __init__(self, R, factors=0.25, n_iter=10, lam=0.001,
                 random_state=None):
        # check the array
        R = check_array(R, dtype=np.float32)  # type: np.ndarray
        n_users, n_items = R.shape

        # get the random state
        random_state = check_random_state(random_state)

        # get the number of factors. If it's a float, compute it
        if isinstance(factors, float):
            factors = min(np.ceil(factors * n_items).astype(int), n_items)

        # the weight matrix is used as a masking matrix when computing the MSE.
        # it allows us to only compute the reconstruction MSE on the rated
        # items, and not the unrated ones.
        W = (R > 0.).astype(np.float32)

        # initialize the first array, Y, and X to None
        Y = random_state.rand(factors, n_items)
        X = None

        # the identity matrix (time lambda) is added to the XX or YY product
        # at each iteration.
        I = np.eye(factors) * lam

        # this list will store all of the training errors
        train_err = []

        # for each iteration, iteratively solve for X, Y, and compute the
        # updated MSE
        for i in xrange(n_iter):
            X = solve(Y.dot(Y.T) + I, Y.dot(R.T)).T
            Y = solve(X.T.dot(X) + I, X.T.dot(R))

            # update the training error
            train_err.append(mse(R, X, Y, W))

        # now we have X, Y, which are our user factors and item factors
        self.X = X
        self.Y = Y
        self.train_err = train_err
        self.n_factors = factors
        self.lam = lam

    def predict(self, R, recompute_users=False):
        """Generate predictions for the test set.

        Computes the predicted product of ``XY`` given the fit factors.
        If recomputing users, will learn the new user factors given the
        existing item factors.
        """
        R = check_array(R, dtype=np.float32, copy=False)  # type: np.ndarray
        Y = self.Y  # item factors
        n_factors, _ = Y.shape

        # we can re-compute user factors on their updated ratings, if we want.
        # (not always advisable, but can be useful for offline recommenders)
        if recompute_users:
            I = np.eye(n_factors) * self.lam
            X = solve(Y.dot(Y.T) + I, Y.dot(R.T)).T
        else:
            X = self.X

        return X.dot(Y)

    def recommend_for_user(self, R, user, n=10, recompute_user=False,
                           filter_previously_seen=False,
                           return_scores=True):
        """Generate predictions for a single user.

        Parameters
        ----------
        R : array-like, shape=(n_users, n_items)
            The test ratings matrix. This must be an explicit ratings matrix
            where 0 indicates an item that a user has not yet rated.

        user : int
            The user index for whom to generate predictions.

        n : int or None, optional (default=10)
            The number of recommendations to return. Default is 10. For all,
            set to None.

        recompute_user : bool, optional (default=False)
            Whether to recompute the user factors given the test set.
            Not always advisable, as it can be considered leakage, but can
            be useful in an offline recommender system where refits are
            infrequent.

        filter_previously_seen : bool, optional (default=False)
            Whether to filter out previously-rated items.

        return_scores : bool, optional (default=True)
            Whether to return the computed scores for the recommended items.

        Returns
        -------
        items : np.ndarray
            The top ``n`` items recommended for the user.

        scores (optional) : np.ndarray
            The corresponding scores for the top ``n`` items for the user.
            Only returned if ``return_scores`` is True.
        """
        R = check_array(R, dtype=np.float32, copy=False)

        # compute the new user vector. Squeeze to make sure it's a vector
        user_vec = self.predict(R, recompute_users=recompute_user)[user, :]
        item_indices = np.arange(user_vec.shape[0])

        # if we are filtering previously seen, remove the prior-rated items
        if filter_previously_seen:
            rated_mask = R[user, :] != 0.
            user_vec = user_vec[~rated_mask]
            item_indices = item_indices[~rated_mask]

        order = np.argsort(-user_vec)[:n]  # descending order of computed scores
        items = item_indices[order]
        if return_scores:
            return items, user_vec[order]
        return items
