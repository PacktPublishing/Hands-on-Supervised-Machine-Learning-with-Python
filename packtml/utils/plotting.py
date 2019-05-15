# -*- coding: utf-8 -*-

from __future__ import absolute_import

from matplotlib.colors import ListedColormap
from matplotlib import pyplot as plt

from packtml.utils.validation import learning_curve

import numpy as np

__all__ = [
    'add_decision_boundary_to_axis',
    'plot_learning_curve'
]


def add_decision_boundary_to_axis(estimator, axis, nclasses,
                                  X_data, stepsize=0.02,
                                  colors=('#FFAAAA', '#AAFFFA', '#AAAAFF')):
    """Plot a classification decision boundary on an axis.

    Estimates lots of values from a classifier and adds the color map
    mesh to an axis. WARNING - use PRIOR to applying scatter values on the
    axis!

    Parameters
    ----------
    estimator : BaseSimpleEstimator
        An estimator that implements ``predict``.

    axis : matplotlib.Axis
        The axis we're plotting on.

    nclasses : int
        The number of classes present in the data

    X_data : np.ndarray, shape=(n_samples, n_features)
        The X data used to fit the data, and along which to plot. Preferably
        2 features for plotting. The first two will be used to plot.

    stepsize : float, optional (default=0.02)
        The size of the steps in the values on which to predict.

    colors : tuple or iterable, optional
        The color map

    Returns
    -------
    xx : np.ndarray
        The x array

    yy : np.ndarray
        The y array

    axis : matplotlib.Axis
        The axis
    """
    x_min, x_max = X_data[:, 0].min() - 1, X_data[:, 0].max() + 1
    y_min, y_max = X_data[:, 1].min() - 1, X_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, stepsize),
                         np.arange(y_min, y_max, stepsize))

    Z = estimator.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    axis.pcolormesh(xx, yy, Z, cmap=ListedColormap(list(colors[:nclasses])))
    return xx, yy, axis


def plot_learning_curve(model, X, y, n_folds, metric, train_sizes,
                        seed=None, trace=False, y_lim=None, **kwargs):
    """Fit and plot a CV learning curve.

    Fits the model with ``n_folds`` of cross-validation over various
    training sizes and computes arrays of scores for the train samples
    and the validation fold samples, then plots them.

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

    y_lim : iterable or None, optional (default=None)
        The y-axis limits

    **kwargs : keyword args or dict
        The keyword args to pass to the estimator.

    Returns
    -------
    plt : Figure
        The matplotlib figure for plotting

    References
    ----------
    .. [1] Based on the scikit-learn example:
           http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
    """
    # delegate the model fits to the function in .validation
    train_scores, val_scores = learning_curve(
        model, X, y, train_sizes=train_sizes,
        metric=metric, seed=seed, trace=trace,
        n_folds=n_folds, **kwargs)

    # compute the means/stds of each scores list
    train_scores_mean = np.mean(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    val_scores_std = np.std(val_scores, axis=1)

    # plot the learning curves
    plt.figure()
    plt.title("Learning curve (model=%s, train sizes=%s)"
              % (model.__name__, str(train_sizes)))

    plt.xlabel("Training sizes")
    plt.ylabel("Score (%s)" % metric.__name__)
    plt.grid()

    # define the y-axis limit if necessary
    if y_lim is not None:
        plt.ylim(y_lim)

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, val_scores_mean - val_scores_std,
                     val_scores_mean + val_scores_std, alpha=0.1,
                     color="g")

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, val_scores_mean, 'o-', color="g",
             label="Validation score")
    plt.legend(loc="best")

    return plt
