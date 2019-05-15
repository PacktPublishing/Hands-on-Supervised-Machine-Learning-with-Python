# -*- coding: utf-8 -*-

from __future__ import absolute_import

from sklearn.utils.validation import check_array
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np

from packtml.recommendation.base import RecommenderMixin
from packtml.base import BaseSimpleEstimator

__all__ = [
    'ItemItemRecommender'
]

try:
    xrange
except NameError:  # py3
    xrange = range


class ItemItemRecommender(BaseSimpleEstimator, RecommenderMixin):
    """Item-to-item recommendation system using cosine similarity.

    A collaborative filtering recommender algorithm that computes the cosine
    similarity between each item and generates recommendations for users'
    highly rated items by returning similar items.

    Parameters
    ----------
    R : array-like, shape=(n_users, n_items)
        The ratings matrix. This must be an explicit ratings matrix where
        0 indicates an item that a user has not yet rated.

    Attributes
    ----------
    similarity : np.ndarray, shape=(n_items, n_items)
        The similarity matrix.

    Notes
    -----
    This implementation is very rudimentary and does not allow tuning of
    hyper-parameters apart from ``k``. No similarity metrics apart from cosine
    similarity may be used. It is largely written to optimize readability. For
    a very highly optimized version, try the "implicit" library.
    """
    def __init__(self, R, k=10):
        # check the array, but don't copy if not needed
        R = check_array(R, dtype=np.float32, copy=False)  # type: np.ndarray

        # save the hyper param for later use later
        self.k = k
        self.similarity = self._compute_sim(R, k)

    def _compute_sim(self, R, k):
        # compute the similarity between all the items. This calculates the
        # similarity between each ITEM
        sim = cosine_similarity(R.T)

        # Only keep the similarities of the top K, setting all others to zero
        # (negative since we want descending)
        not_top_k = np.argsort(-sim, axis=1)[:, k:]  # shape=(n_items, k)

        if not_top_k.shape[1]:  # only if there are cols (k < n_items)
            # now we have to set these to zero in the similarity matrix
            row_indices = np.repeat(range(not_top_k.shape[0]),
                                    not_top_k.shape[1])
            sim[row_indices, not_top_k.ravel()] = 0.

        return sim

    def recommend_for_user(self, R, user, n=10,
                           filter_previously_seen=False,
                           return_scores=True, **kwargs):
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

        filter_previously_seen : bool, optional (default=False)
            Whether to filter out previously-rated items.

        return_scores : bool, optional (default=True)
            Whether to return the computed scores for the recommended items.

        **kwargs : keyword args
            Ignored. Present to match super signature.

        Returns
        -------
        items : np.ndarray
            The top ``n`` items recommended for the user.

        recommendations (optional) : np.ndarray
            The corresponding scores for the top ``n`` items for the
            user. Only returned if ``return_scores`` is True.
        """

        # check the array and get the user vector
        R = check_array(R, dtype=np.float32, copy=False)
        user_vector = R[user, :]

        # compute the dot product between the user vector and the similarity
        # matrix
        recommendations = user_vector.dot(self.similarity)  # shape=(n_items,)

        # if we're filtering previously-seen items, now is the time to do that
        item_indices = np.arange(recommendations.shape[0])
        if filter_previously_seen:
            rated_mask = user_vector != 0.
            recommendations = recommendations[~rated_mask]
            item_indices = item_indices[~rated_mask]

        # now arg sort descending (most similar items first)
        order = np.argsort(-recommendations)[:n]
        items = item_indices[order]

        if return_scores:
            return items, recommendations[order]
        return items

    def predict(self, R):
        """Generate predictions for the test set.

        Computes the predicted product of users' rated vectors on the
        pre-computed similarity matrix.
        """
        R = check_array(R, dtype=np.float32, copy=False)  # type: np.ndarray

        # compute the product R*sim
        return R.dot(self.similarity)
