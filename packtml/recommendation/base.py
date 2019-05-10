# -*- coding: utf-8 -*-

from __future__ import absolute_import

import six
from abc import ABCMeta, abstractmethod

__all__ = [
    'RecommenderMixin'
]

try:
    xrange
except NameError:  # py3
    xrange = range


class RecommenderMixin(six.with_metaclass(ABCMeta)):
    """Mixin interface for recommenders.

    This class should be inherited by recommender algorithms. It provides an
    abstract interface for generating recommendations for a user, and a
    function for creating recommendations for all users.
    """
    @abstractmethod
    def recommend_for_user(self, R, user, n=10, filter_previously_seen=False,
                           return_scores=True, **kwargs):
        """Generate recommendations for a user.

        A method that should be overridden by subclasses to create
        recommendations via their own prediction strategy.
        """

    def recommend_for_all_users(self, R, n=10,
                                filter_previously_seen=False,
                                return_scores=True, **kwargs):
        """Create recommendations for all users."""
        return (
            self.recommend_for_user(
                R, user, n=n, filter_previously_seen=filter_previously_seen,
                return_scores=return_scores, **kwargs)
            for user in xrange(R.shape[0]))
