# -*- coding: utf-8 -*-
#
# Author: Taylor G Smith <taylor.smith@alkaline-ml.com>
#
# A simplified version of Classification and Regression Trees. This file
# is intended to maximize readability and understanding of how CART trees work.
# For very fast or customizable decision tree solutions, use scikit-learn.
#
# The best order in which to read & understand the contents to best
# grok the entire concept:
#
#   1. metrics.InformationGain & metrics.VarianceReduction
#   2. RandomSplitter
#   3. LeafNode
#   4. BaseCART

from __future__ import absolute_import, division

from sklearn.utils.validation import check_X_y, check_random_state, check_array
from sklearn.utils.multiclass import check_classification_targets
from sklearn.base import ClassifierMixin, RegressorMixin, is_classifier

import numpy as np

from packtml.base import BaseSimpleEstimator
from packtml.decision_tree.metrics import InformationGain, VarianceReduction

__all__ = [
    'CARTRegressor',
    'CARTClassifier'
]

try:
    xrange
except NameError:  # py3
    xrange = range


class RandomSplitter(object):
    """Evaluate a split via random values in a feature.

    Every feature in the dataset needs to be evaluated in a CART tree. Since
    that in itself can be expensive, the random splitter allows us to look at
    only a random amount of row splits per feature in order to make the best
    splitting decision.

    Parameters
    ----------
    random_state : np.random.RandomState
        The random state for seeding the choices

    criterion : callable
        The metric used for evaluating the "goodness" of a split. Either
        ``InformationGain`` (with entropy or Gini) for classification, or
        ``VarianceReduction`` for regression.

    n_val_sample : float, optional (default=25)
        The number of values per feature to sample as a splitting point.
    """
    def __init__(self, random_state, criterion, n_val_sample=25):
        self.random_state = random_state
        self.criterion = criterion  # BaseCriterion from metrics
        self.n_val_sample = n_val_sample

    def find_best(self, X, y):
        criterion = self.criterion
        rs = self.random_state

        # keep track of the best info gain
        best_gain = 0.

        # keep track of best feature and best value on which to split
        best_feature = None
        best_value = None

        # get the current state of the uncertainty (gini or entropy)
        uncertainty = criterion.compute_uncertainty(y)

        # iterate over each feature
        for col in xrange(X.shape[1]):
            feature = X[:, col]

            # get all values in the feature
            # values = np.unique(feature)
            seen_values = set()

            # the number of values to sample. Should be defined as the min
            # between the prescribed n_val_sample value and the number of
            # unique values in the feature.
            n_vals = min(self.n_val_sample, np.unique(feature).shape[0])

            # For each of n_val_sample iterations, select a random value
            # from the feature and create a split. We store whether we've seen
            # the value before; if we have, continue. Continue until we've seen
            # n_vals unique values. This allows us to more likely select values
            # that are high frequency (retains distributional data implicitly)
            for v in rs.permutation(feature):

                # if we've hit the limit of the number of values we wanted to
                # examine, break out
                if len(seen_values) == n_vals:
                    break

                # if we've already tried this value, continue
                elif v in seen_values:  # O(1) lookup
                    continue

                # otherwise, it's a new value we've never tried splitting on.
                # add it to the set.
                seen_values.add(v)

                # create the mask (these values "go left")
                mask = feature >= v  # type: np.ndarray

                # skip this step if this doesn't divide the dataset
                if np.unique(mask).shape[0] == 1:  # all True or all False
                    continue

                # compute how good this split was
                gain = criterion(y, mask, uncertainty=uncertainty)

                # if the gain is better, we keep this feature & value &
                # update the best gain we've seen so far
                if gain > best_gain:
                    best_feature = col
                    best_value = v
                    best_gain = gain

        # if best feature is None, it means we never found a viable split...
        # this is likely because all of our labels were perfect. In this case,
        # we could select any feature and the first value and define that as
        # our left split and nothing will go right.
        if best_feature is None:
            best_feature = 0
            best_value = np.squeeze(X[:, best_feature])[0]
            best_gain = 0.

        # we need to know the best feature, the best value, and the best gain
        return best_feature, best_value, best_gain


class LeafNode(object):
    """A tree node class.

    Tree node that store the column on which to split and the value above
    which to go left vs. right. Additionally, it stores the target statistic
    related to this node. For instance, in a classification scenario:

        >>> X = np.array([[ 1, 1.5 ],
        ...               [ 2, 0.5 ],
        ...               [ 3, 0.75]])
        >>> y = np.array([0, 1, 1])
        >>> node = LeafNode(split_col=0, split_val=2,
        ...                 class_statistic=_most_common(y))

    This means if ``node`` were a terminal node, it would generate predictions
    of 1, since that was the most common value in the pre-split ``y``. The
    class statistic will differ for splits in the tree, where the most common
    value in ``y`` for records in ``X`` that go left is 1, and 0 for that which
    goes to the right.

    The class statistic is computed for each split as the tree recurses.

    Parameters
    ----------
    split_col : int
        The column on which to split.

    split_val : float or int
        The value above which to go left.
    """
    def __init__(self, split_col, split_val, split_gain, class_statistic):

        self.split_col = split_col
        self.split_val = split_val
        self.split_gain = split_gain

        # the class statistic is the mode or the mean of the targets for
        # this split
        self.class_statistic = class_statistic

        # if these remain None, it's a terminal node
        self.left = None
        self.right = None

    def create_split(self, X, y):
        """Split the next X, y.

        Returns
        -------
        X_left : np.ndarray, shape=(n_samples, n_features)
            Rows where ``split_col >= split_val``.

        X_right : np.ndarray, shape=(n_samples, n_features)
            Rows where ``split_col < split_val``.

        y_left : np.ndarray, shape=(n_samples,)
            Target where ``split_col >= split_val``.

        y_right : np.ndarray, shape=(n_samples,)
            Target where ``split_col < split_val``.
        """
        # If values in the split column are greater than or equal to the
        # split value, we go left.
        left_mask = X[:, self.split_col] >= self.split_val

        # Otherwise we go to the right
        right_mask = ~left_mask  # type: np.ndarray

        # If the left mask is all False or all True, it means we've achieved
        # a perfect split.
        all_left = left_mask.all()
        all_right = right_mask.all()

        # create the left split. If it's all right side, we'll return None
        X_left = X[left_mask, :] if not all_right else None
        y_left = y[left_mask] if not all_right else None

        # create the right split. If it's all left side, we'll return None.
        X_right = X[right_mask, :] if not all_left else None
        y_right = y[right_mask] if not all_left else None

        return X_left, X_right, y_left, y_right

    def is_terminal(self):
        """Determine whether the node is terminal.

        If there is no left node and no right node, it's a terminal node.
        If either is non-None, it is a parent to something.
        """
        return self.left is None and self.right is None

    def __repr__(self):
        """Get the string representation of the node."""
        return "Rule: Go left if x%i >= %r else go right (gain=%.3f)" \
               % (self.split_col, self.split_val, self.split_gain)

    def predict_record(self, record):
        """Find the terminal node in the tree and return the class statistic"""
        # First base case, this is a terminal node:
        has_left = self.left is not None
        has_right = self.right is not None
        if not has_left and not has_right:
            return self.class_statistic

        # Otherwise, determine whether the record goes right or left
        go_left = record[self.split_col] >= self.split_val

        # if we go left and there is a left node, delegate the recursion to the
        # left side
        if go_left and has_left:
            return self.left.predict_record(record)

        # if we go right, delegate to the right
        if not go_left and has_right:
            return self.right.predict_record(record)

        # if we get here, it means one of two things:
        # 1. we were supposed to go left and didn't have a left
        # 2. we were supposed to go right and didn't have a right
        # for both of these, we return THIS class statistic
        return self.class_statistic


def _most_common(y):
    # This is essentially just a "mode" function to compute the most
    # common value in a vector.
    cls, cts = np.unique(y, return_counts=True)
    order = np.argsort(-cts)
    return cls[order][0]


class _BaseCART(BaseSimpleEstimator):
    def __init__(self, X, y, criterion, min_samples_split, max_depth,
                 n_val_sample, random_state):
        # make sure max_depth > 1
        if max_depth < 2:
            raise ValueError("max depth must be > 1")

        # check the input arrays, and if it's classification validate the
        # target values in y
        X, y = check_X_y(X, y, accept_sparse=False, dtype=None, copy=True)
        if is_classifier(self):
            check_classification_targets(y)

        # hyper parameters so we can later inspect attributes of the model
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_val_sample = n_val_sample
        self.random_state = random_state

        # create the splitting class
        random_state = check_random_state(random_state)
        self.splitter = RandomSplitter(random_state, criterion, n_val_sample)

        # grow the tree depth first
        self.tree = self._find_next_split(X, y, 0)

    def _target_stat(self, y):
        """Given a vector, ``y``, decide what value to return as the leaf
        node statistic (mean for regression, mode for classification)
        """

    def _find_next_split(self, X, y, current_depth):
        # base case 1: current depth is the limit, the parent node should
        # be a terminal node (child = None)
        # base case 2: n samples in X <= min_samples_split
        if current_depth == self.max_depth or \
                X.shape[0] <= self.min_samples_split:
            return None

        # create the next split
        split_feature, split_value, gain = \
            self.splitter.find_best(X, y)

        # create the next node based on the best split feature and value
        # that we just found. Also compute the "target stat" (mode of y for
        # classification problems or mean of y for regression problems) and
        # pass that to the node in case it is the terminal node (i.e., the
        # decision maker)
        node = LeafNode(split_feature, split_value, gain, self._target_stat(y))

        # Create the splits based on the criteria we just determined, and then
        # recurse down left, right sides
        X_left, X_right, y_left, y_right = node.create_split(X, y)

        # if either the left or right is None, it means we've achieved a
        # perfect split. It is then a terminal node and will remain None.
        if X_left is not None:
            node.left = self._find_next_split(X_left, y_left,
                                              current_depth + 1)

        if X_right is not None:
            node.right = self._find_next_split(X_right, y_right,
                                               current_depth + 1)

        return node

    def predict(self, X):
        # Check the array
        X = check_array(X, dtype=np.float32)  # type: np.ndarray

        # For each record in X, find its leaf node in the tree (O(log N))
        # to get the predictions. This makes the prediction operation
        # O(N log N) runtime complexity
        predictions = [self.tree.predict_record(row) for row in X]
        return np.asarray(predictions)


class CARTRegressor(_BaseCART, RegressorMixin):
    """Decision tree regression.

    Builds a decision tree to solve a regression problem using the CART
    algorithm. The estimator builds a binary tree structure, evaluating each
    feature at each iteration to recursively split along the best value and
    progress down the tree until each leaf node reaches parsimony.

    The regression tree uses "variance reduction" to assess the "goodness"
    of a split, selecting the split and feature that maximizes the value.

    To make predictions, each record is evaluated at each node of the tree
    until it reaches a leaf node. For regression, predictions are made by
    returning the training target's mean for the leaf node.

    Parameters
    ----------
    X : array-like, shape=(n_samples, n_features)
        The training array. Should be a numpy array or array-like structure
        with only finite values.

    y : array-like, shape=(n_samples,)
        The target vector.

    max_depth : int, optional (default=5)
        The maximum depth to which the tree will grow. Note that the tree is
        not guaranteed to reach this depth and may stop growing early if the
        ``min_samples_split`` terminal criterion is met first.

    min_samples_split : int, optional (default=1)
        A terminal criterion used to halt the growth of a tree. If a leaf
        node's split contains <= ``min_samples_split``, it will not grow
        any further.

    n_val_sample : int, optional (default=25)
        The method by which we evaluate splits differs a bit from highly
        optimized libraries like scikit-learn, which may evaluate for the
        globally optimal split for each feature. We use random splitting
        which evaluates a number of unique values for each feature at each
        split. The ``n_val_sample`` is the maximum number of values per
        feature that will be evaluated as a potential splitting point at
        each iteration.

    random_state : int, None or RandomState, optional (default=None)
        The random state used to seed the RandomSplitter.

    Attributes
    ----------
    splitter : RandomSplitter
        The feature splitting class. Used for determining optimal splits at
        each node.

    tree : LeafNode
        The actual tree. Each node contains data on the class statistic (i.e.,
        mode or mean of the training target at that split), best feature and
        best value.
    """
    def __init__(self, X, y, max_depth=5, min_samples_split=1,
                 n_val_sample=25, random_state=None):

        super(CARTRegressor, self).__init__(
            X, y, criterion=VarianceReduction(),
            min_samples_split=min_samples_split, max_depth=max_depth,
            n_val_sample=n_val_sample, random_state=random_state)

    def _target_stat(self, y):
        """Given a vector, ``y``, get the mean"""
        return y.mean()


class CARTClassifier(_BaseCART, ClassifierMixin):
    """Decision tree classication.

    Builds a decision tree to solve a classification problem using the CART
    algorithm. The estimator builds a binary tree structure, evaluating each
    feature at each iteration to recursively split along the best value and
    progress down the tree until each leaf node reaches parsimony.

    The classification tree uses "information gain" to assess the "goodness"
    of a split, selecting the split and feature that maximizes the value.

    To make predictions, each record is evaluated at each node of the tree
    until it reaches a leaf node. For classification, predictions are made by
    returning the training target's mode for the leaf node.

    Parameters
    ----------
    X : array-like, shape=(n_samples, n_features)
        The training array. Should be a numpy array or array-like structure
        with only finite values.

    y : array-like, shape=(n_samples,)
        The target vector.

    criterion : str or unicode, optional (default='gini')
        The splitting criterion used for classification problems. CART trees
        typically use "gini" but their cousins, C4.5 trees, use "entropy". Both
        metrics are extremely similar and will likely not change your tree
        structure by much.

    max_depth : int, optional (default=5)
        The maximum depth to which the tree will grow. Note that the tree is
        not guaranteed to reach this depth and may stop growing early if the
        ``min_samples_split`` terminal criterion is met first.

    min_samples_split : int, optional (default=1)
        A terminal criterion used to halt the growth of a tree. If a leaf
        node's split contains <= ``min_samples_split``, it will not grow
        any further.

    n_val_sample : int, optional (default=25)
        The method by which we evaluate splits differs a bit from highly
        optimized libraries like scikit-learn, which may evaluate for the
        globally optimal split for each feature. We use random splitting
        which evaluates a number of unique values for each feature at each
        split. The ``n_val_sample`` is the maximum number of values per
        feature that will be evaluated as a potential splitting point at
        each iteration.

    random_state : int, None or RandomState, optional (default=None)
        The random state used to seed the RandomSplitter.

    Attributes
    ----------
    splitter : RandomSplitter
        The feature splitting class. Used for determining optimal splits at
        each node.

    tree : LeafNode
        The actual tree. Each node contains data on the class statistic (i.e.,
        mode or mean of the training target at that split), best feature and
        best value.
    """
    def __init__(self, X, y, criterion='gini', max_depth=5,
                 min_samples_split=1, n_val_sample=25, random_state=None):

        super(CARTClassifier, self).__init__(
            X, y, criterion=InformationGain(criterion), max_depth=max_depth,
            min_samples_split=min_samples_split,
            n_val_sample=n_val_sample, random_state=random_state)

    def _target_stat(self, y):
        """Given a vector, ``y``, get the mode"""
        return _most_common(y)
