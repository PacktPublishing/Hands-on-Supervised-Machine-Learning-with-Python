# -*- coding: utf-8 -*-

from __future__ import absolute_import

from packtml.decision_tree.metrics import (entropy, gini_impurity,
                                           InformationGain)

import numpy as np
from numpy.testing import assert_almost_equal


def test_entropy():
    events = np.asarray(9 * [0] + 5 * [1])  # 9/14, 5/14
    ent = entropy(events)
    assert round(ent, 2) == 0.94, round(ent, 2)


def test_gini_impurity():
    x = np.asarray([0] * 10 + [1] * 10)
    assert gini_impurity(x) == 0.5
    assert gini_impurity(x[:10]) == 0.

    # show that no mixing of gini yields 0.0
    assert gini_impurity(np.array([0, 0])) == 0.

    # with SOME mixing we get 0.5
    assert gini_impurity(np.array([0, 1])) == 0.5

    # with a lot of mixing we get a number close to 0.8
    gi = gini_impurity([0, 1, 2, 3, 4])
    assert_almost_equal(gi, 0.8)


def test_information_gain():
    X = np.array([
        [0, 3],
        [1, 3],
        [2, 1],
        [2, 1],
        [1, 3]
    ])

    y = np.array([0, 0, 1, 1, 2])

    uncertainty = gini_impurity(y)
    assert_almost_equal(uncertainty, 0.63999999)
    mask = X[:, 0] == 0

    # compute the info gain for this mask
    infog = InformationGain("gini")
    ig = infog(y, mask, uncertainty)
    assert_almost_equal(ig, 0.1399999)
