# -*- coding: utf-8 -*-

from __future__ import absolute_import

from numpy.testing import assert_array_equal, assert_almost_equal
import numpy as np

from packtml.decision_tree.metrics import InformationGain
from packtml.decision_tree.cart import (CARTClassifier, CARTRegressor,
                                        RandomSplitter, LeafNode, _most_common)

X = np.array([[0, 1, 2],
              [1, 2, 3],
              [2, 3, 4]])

y = np.array([0, 1, 1])

X2 = np.array([[0, 1, 2],
               [1, 2, 3],
               [2, 3, 4],
               [3, 4, 5],
               [4, 5, 6],
               [5, 6, 7]])

y2 = np.array([0, 0, 1, 1, 1, 1])

# a regression dataset
rs = np.random.RandomState(42)
Xreg = np.sort(5 * rs.rand(100, 1), axis=0)
yreg = np.sin(Xreg).ravel()


def test_most_common():
    assert _most_common(y) == 1
    assert _most_common([1]) == 1


def test_terminal_leaf_node():
    node = LeafNode(split_col=0, split_val=1.,
                    class_statistic=_most_common(y),
                    split_gain=np.inf)

    # show that there are no children
    assert node.is_terminal()

    # show that the splitting works as expected
    X_left, X_right, y_left, y_right = node.create_split(X, y)
    assert_array_equal(X_left, X[1:, :])
    assert_array_equal(X_right, X[:1, :])
    assert_array_equal(y_left, [1, 1])
    assert_array_equal(y_right, [0])

    # show that predictions work as expected
    assert [node.predict_record(r) for r in X] == [1, 1, 1]


def test_complex_leaf_node():
    node = LeafNode(split_col=0, split_val=3.,
                    class_statistic=_most_common(y2),
                    split_gain=np.inf)

    # create the split
    X_left, X_right, y_left, y_right = node.create_split(X2, y2)

    # show it worked as expected
    assert_array_equal(X_left, X2[3:, :])
    assert_array_equal(X_right, X2[:3, :])
    assert_array_equal(y_left, [1, 1, 1])
    assert_array_equal(y_right, [0, 0, 1])

    # show that if we CURRENTLY predicted on the bases of node being the
    # terminal leaf, we'd get all 1s.
    get_preds = (lambda: [node.predict_record(r) for r in X2])
    assert get_preds() == [1, 1, 1, 1, 1, 1]

    # add a sub node to the right side
    right_node = LeafNode(split_col=0, split_val=2.,
                          class_statistic=_most_common(y_right),
                          split_gain=np.inf)

    assert right_node.class_statistic == 0.

    # attach to the original node and assert it's not terminal anymore
    node.right = right_node
    assert not node.is_terminal()

    # now our predictions should differ!
    assert get_preds() == [0, 0, 0, 1, 1, 1]


def test_fit_classifier():
    # show we can fit a classifier
    clf = CARTClassifier(X, y)
    # show we can predict
    clf.predict(X)


def test_fit_regressor():
    # show we can fit a regressor
    reg = CARTRegressor(Xreg, yreg)
    # show we can predict
    reg.predict(Xreg)


def test_random_splitter():
    pre_X = np.array([[21, 3], [4, 2], [37, 2]])
    pre_y = np.array([1, 0, 1])

    # this is the splitting class; we'll use gini as the criteria
    random_state = np.random.RandomState(42)
    splitter = RandomSplitter(random_state=random_state,
                              criterion=InformationGain('gini'),
                              n_val_sample=3)

    # find the best:
    best_feature, best_value, best_gain = splitter.find_best(pre_X, pre_y)
    assert best_feature == 0
    assert best_value == 21
    assert_almost_equal(best_gain, 0.4444444444, decimal=8)
