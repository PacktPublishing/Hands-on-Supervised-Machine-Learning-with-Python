# -*- coding: utf-8 -*-

from __future__ import absolute_import

from abc import ABCMeta, abstractmethod
import six

__all__ = [
    'BaseSimpleEstimator'
]


class BaseSimpleEstimator(six.with_metaclass(ABCMeta)):
    """Base class for packt estimators.

    The estimators in the Packt package do not behave exactly like scikit-learn
    estimators (by design). They are made to perform the model fit immediately
    upon class instantiation. Moreover, many of the hyper-parameter options
    are limited to promote readability and avoid confusion.

    The constructor of every Packt estimator should resemble the following::

        def __init__(self, X, y, *args, **kwargs):
            ...

    where ``X`` is the training matrix, ``y`` is the training target variable,
    and ``*args`` and ``**kwargs`` are varargs that will differ for each
    estimator.
    """
    @abstractmethod
    def predict(self, X):
        """Form predictions based on new data.

        This function must be implemented by subclasses to generate
        predictions given the model fit.

        Parameters
        ----------
        X : array-like, shape=(n_samples, n_features)
            The test array. Should be only finite values.
        """
