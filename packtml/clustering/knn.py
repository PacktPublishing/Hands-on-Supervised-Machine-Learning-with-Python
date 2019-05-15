# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>
#
# An implementation of kNN clustering. Note that this was written to
# maximize readability. To use kNN in a true project setting, you may
# wish to use a more highly optimized library, such as scikit-learn.

from __future__ import absolute_import

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils.validation import check_X_y
from sklearn.utils.multiclass import check_classification_targets

from scipy.stats import mode
import numpy as np

from packtml.base import BaseSimpleEstimator

__all__ = [
    'KNNClassifier'
]


class KNNClassifier(BaseSimpleEstimator):
    """Classify points using k-Nearest Neighbors.

    The kNN algorithm computes the distances between points in a matrix and
    identifies the nearest "neighboring" points to each observation. The idea
    is that neighboring points share similar attributes. Therefore, if a
    neighbor is of some class, an unknown observation may likely belong to
    the same class.

    There are several caveats to kNN:

        * We have to retain all of the training data, which is expensive.
        * Computing the pairwise distance matrix is also expensive.
        * You should make sure you've standardized your data (mean 0, stddev 1)
          prior to fitting a kNN model

    Parameters
    ----------
    X : array-like, shape=(n_samples, n_features)
        The training array. Should be a numpy array or array-like structure
        with only finite values.

    y : array-like, shape=(n_samples,)
        The target vector.

    k : int, optional (default=10)
        The number of neighbors to identify. The higher the ``k`` parameter,
        the more likely you are to *under*-fit your data. The lower the ``k``
        parameter, the more likely you are to *over*-fit your model.

    Notes
    -----
    This is a very rudimentary implementation of KNN. It does not permit tuning
    of distance metrics, optimization of the search algorithm or any other
    parameters. It is written to be as simple as possible to maximize
    readability. For a more optimal solution, see
    ``sklearn.neighbors.KNeighborsClassifier``.
    """
    def __init__(self, X, y, k=10):
        # check the input array
        X, y = check_X_y(X, y, accept_sparse=False, dtype=np.float32,
                         copy=True)

        # make sure we're performing classification here
        check_classification_targets(y)

        # Save the K hyper-parameter so we can use it later
        self.k = k

        # kNN is a special case where we have to save the training data in
        # order to make predictions in the future
        self.X = X
        self.y = y

    def predict(self, X):
        # Compute the pairwise distances between each observation in
        # the dataset and the training data. This can be relatively expensive
        # for very large datasets!!
        train = self.X
        dists = euclidean_distances(X, train)

        # Arg sort to find the shortest distance for each row. This sorts
        # elements in each row (independent of other rows) to determine the
        # order required to sort the rows.
        # I.e:
        # >>> P = np.array([[4, 5, 1], [3, 1, 6]])
        # >>> np.argsort(P, axis=1)
        # array([[2, 0, 1],
        #        [1, 0, 2]])
        nearest = np.argsort(dists, axis=1)

        # We only care about the top K, really, so get sorted and then truncate
        # I.e:
        # array([[1, 2, 1],
        #           ...
        #        [0, 0, 0]])
        predicted_labels = self.y[nearest][:, :self.k]

        # We want the most common along the rows as the predictions
        # I.e:
        # array([1, ..., 0])
        return mode(predicted_labels, axis=1)[0].ravel()
