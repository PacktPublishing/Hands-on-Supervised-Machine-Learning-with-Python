# -*- coding: utf-8 -*-

from __future__ import absolute_import

from sklearn.externals import six
from sklearn.model_selection import ShuffleSplit

import numpy as np

__all__ = [
    'assert_is_binary',
    'is_iterable',
    'learning_curve'
]


def assert_is_binary(y):
    """Validate that a vector is binary.

    Checks that a vector is binary. This utility is used by all of
    the simple classifier estimators to validate the input target.

    Parameters
    ----------
    y : np.ndarray, shape=(n_samples,)
        The target vector
    """
    # validate that y is in (0, 1)
    unique_y = np.unique(y)  # type: np.ndarray
    if unique_y.shape[0] != 2 or [0, 1] != unique_y.tolist():
        raise ValueError("y must be binary, but got unique values of %s"
                         % str(unique_y))


def is_iterable(x):
    """Determine whether an item is iterable.

    Python 3 introduced the ``__iter__`` functionality to
    strings, making them falsely behave like iterables. This
    function determines whether an object is an iterable given
    the presence of the ``__iter__`` method and that the object
    is *not* a string.

    Parameters
    ----------
    x : int, object, str, iterable, None
        The object in question. Could feasibly be any type.
    """
    if isinstance(x, six.string_types):
        return False
    return hasattr(x, "__iter__")


def learning_curve(model, X, y, metric, train_sizes, n_folds=3,
                   seed=None, trace=False, **kwargs):
    """Fit a CV learning curve.

    Fits the model with ``n_folds`` of cross-validation over various
    training sizes and returns arrays of scores for the train samples
    and the validation fold samples.

    Parameters
    ----------
    model : BaseSimpleEstimator
        The model class that should be fit.

    X : array-like, shape=(n_samples, n_features)
        The training matrix.

    y : array-like, shape=(n_samples,)
        The training labels/ground-truth.

    metric : callable
        The scoring metric

    train_sizes : iterable
        The size of the training set for each fold.

    n_folds : int, optional (default=3)
        The number of CV folds

    seed : int or None, optional (default=None)
        The random seed for cross validation.

    trace : bool, optional (default=False)
        Whether to print to stdout after each set of folds is fit
        for a given train size.

    **kwargs : keyword args or dict
        The keyword args to pass to the estimator.

    Returns
    -------
    train_scores : np.ndarray, shape=(n_trials, n_folds)
        The scores for the train samples. Each row represents a
        trial (new train size), and each column corresponds to the
        fold of the trial, i.e., for ``n_folds=3``, there will be
        3 columns.

    val_scores : np.ndarray, shape=(n_trials, n_folds)
        The scores for the validation folds. Each row represents a
        trial (new train size), and each column corresponds to the
        fold of the trial, i.e., for ``n_folds=3``, there will be
        3 columns.
    """
    # Each of these lists will be a 2d array. A row will represent a
    # trial for a particular train size, and each column will
    # correspond with a fold.
    train_scores = []
    val_scores = []

    # The number of samples in the dataset
    n_samples = X.shape[0]

    # If the input is a pandas frame, make it a numpy array for indexing
    if hasattr(X, "iloc"):
        X = X.values

    # We need to validate that all of the sizes within the train_sizes
    # are less than the number of samples in the dataset!
    assert all(s < n_samples for s in train_sizes), \
        "All train sizes (%s) must be less than n_samples (%i)" \
        % (str(train_sizes), n_samples)

    # For each training size, we're going to initialize a new KFold
    # cross validation instance and fit the K folds...
    for train_size in train_sizes:
        cv = ShuffleSplit(n_splits=n_folds,
                          train_size=train_size,
                          test_size=n_samples - train_size,
                          random_state=seed)

        # This is the inner list (row) that will represent the
        # scores for this train size
        inner_train_scores = []
        inner_val_scores = []

        # get our splits
        for train_indices, test_indices in cv.split(X, y):
            # get the training samples
            train_X = X[train_indices, :]
            train_y = y.take(train_indices)

            # fit the model
            m = model(train_X, train_y, **kwargs)

            # score the model on the train set
            inner_train_scores.append(
                metric(train_y, m.predict(train_X)))

            # score the model on the validation set
            inner_val_scores.append(
                metric(y.take(test_indices),
                       m.predict(X[test_indices, :])))

        # Now attach the inner lists to the outer lists
        train_scores.append(inner_train_scores)
        val_scores.append(inner_val_scores)

        if trace:
            print("Completed fitting %i folds for train size=%i"
                  % (n_folds, train_size))

    # Make our train/val arrays into numpy arrays
    train_scores = np.asarray(train_scores)
    val_scores = np.asarray(val_scores)

    return train_scores, val_scores
