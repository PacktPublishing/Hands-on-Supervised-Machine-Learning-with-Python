# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>
#
# Metrics used for determining how to split a feature in a decision tree.

from __future__ import absolute_import

import numpy as np

__all__ = [
    'entropy',
    'gini_impurity',
    'InformationGain',
    'VarianceReduction'
]


def _clf_metric(y, metric):
    """Internal helper. Since this is internal, so no validation performed"""
    # get unique classes in y
    y = np.asarray(y)
    C, cts = np.unique(y, return_counts=True)

    # a base case is that there is only one class label
    if C.shape[0] == 1:
        return 0.

    pr_C = cts.astype(float) / y.shape[0]  # P(Ci)

    # 1 - sum(P(Ci)^2)
    if metric == 'gini':
        return 1. - pr_C.dot(pr_C)  # np.sum(pr_C ** 2)
    elif metric == 'entropy':
        return np.sum(-pr_C * np.log2(pr_C))

    # shouldn't ever get to this point since it is internal
    else:
        raise ValueError("metric should be one of ('gini', 'entropy'), "
                         "but encountered %s" % metric)


def entropy(y):
    """Compute the entropy of class labels.

    This computes the entropy of training samples. A high entropy means
    a relatively uniform distribution, while low entropy indicates a
    varying distribution (many peaks and valleys).

    References
    ----------
    .. [1] http://www.cs.csi.cuny.edu/~imberman/ai/Entropy%20and%20Information%20Gain.htm
    """
    return _clf_metric(y, 'entropy')


def gini_impurity(y):
    """Compute the Gini index on a target variable.

    The Gini index gives an idea of how mixed two classes are within a leaf
    node. A perfect class separation will result in a Gini impurity of 0 (i.e.,
    "perfectly pure").
    """
    return _clf_metric(y, 'gini')


class BaseCriterion(object):
    """Splitting criterion.

    Base class for InformationGain and VarianceReduction. WARNING - do
    not invoke this class directly. Use derived classes only! This is a
    loosely-defined abstract class used to prescribe a common interface
    for sub-classes.
    """
    def compute_uncertainty(self, y):
        """Compute the uncertainty for a vector.

        A subclass should override this function to compute the uncertainty
        (i.e., entropy or gini) of a vector.
        """


class InformationGain(BaseCriterion):
    """Compute the information gain after a split.

    The information gain metric is used by CART trees in a classification
    context. It measures the difference in the gini or entropy before and
    after a split to determine whether the split "taught" us anything.

    Parameters
    ----------
    metric : str or unicode
        The name of the metric to use. Either "gini" (Gini impurity)
        or "entropy".
    """
    def __init__(self, metric):
        # let fail out with a KeyError if an improper metric
        self.crit = {'gini': gini_impurity,
                     'entropy': entropy}[metric]

    def compute_uncertainty(self, y):
        """Compute the uncertainty for a vector.

        This method computes either the Gini impurity or entropy of a target
        vector using the prescribed method.
        """
        return self.crit(y)

    def __call__(self, target, mask, uncertainty):
        """Compute the information gain of a split.

        Parameters
        ----------
        target : np.ndarray
            The target feature

        mask : np.ndarray
            The value mask

        uncertainty : float
            The gini or entropy of rows pre-split
        """
        left, right = target[mask], target[~mask]
        p = float(left.shape[0]) / float(target.shape[0])

        crit = self.crit  # type: callable
        return uncertainty - p * crit(left) - (1 - p) * crit(right)


class VarianceReduction(BaseCriterion):
    """Compute the variance reduction after a split.

    Variance reduction is a splitting criterion used by CART trees in the
    context of regression. It examines the variance in a target before and
    after a split to determine whether we've reduced the variability in the
    target.
    """
    def compute_uncertainty(self, y):
        """Compute the variance of a target."""
        return np.var(y)

    def __call__(self, target, mask, uncertainty):
        left, right = target[mask], target[~mask]
        return uncertainty - (self.compute_uncertainty(left) +
                              self.compute_uncertainty(right))
