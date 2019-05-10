# -*- coding: utf-8 -*-

from __future__ import absolute_import

import six
from abc import ABCMeta, abstractmethod

import numpy as np

__all__ = [
    'tanh',
    'NeuralMixin'
]


def tanh(X):
    """Hyperbolic tangent.

    Compute the tan-h (Hyperbolic tangent) activation function.
    This is a very easily-differentiable activation function.

    Parameters
    ----------
    X : np.ndarray, shape=(n_samples, n_features)
        The transformed X array (X * W + b).
    """
    return np.tanh(X)


class NeuralMixin(six.with_metaclass(ABCMeta)):
    """Abstract interface for neural network classes."""
    @abstractmethod
    def export_weights_and_biases(self, output_layer=True):
        """Return the weights and biases of the network"""
