# -*- coding: utf-8 -*-

from __future__ import absolute_import

from packtml.clustering import KNNClassifier

from sklearn.datasets import load_iris
from numpy.testing import assert_array_equal
import numpy as np

iris = load_iris()
X = iris.data[:, :2]
y = iris.target


def test_knn():
    # show we can fit
    knn = KNNClassifier(X, y)
    # show we can predict
    knn.predict(X)


def test_knn2():
    X2 = np.array([[0., 0., 0.5],
                   [0., 0.5, 0.],
                   [0.5, 0., 0.],
                   [5., 5., 6.],
                   [6., 5., 5.]])

    y2 = [0, 0, 0, 1, 1]
    knn = KNNClassifier(X2, y2, k=3)
    preds = knn.predict(X2)
    assert_array_equal(preds, y2)
