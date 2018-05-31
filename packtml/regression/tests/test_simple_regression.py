# -*- coding: utf-8 -*-

from __future__ import absolute_import

from packtml.regression import SimpleLinearRegression

import numpy as np
from numpy.testing import assert_almost_equal


def test_simple_linear_regression():
    # y = 2a + 1.5b + 0
    random_state = np.random.RandomState(42)
    X = random_state.rand(100, 2)
    y = 2. * X[:, 0] + 1.5 * X[:, 1]

    lm = SimpleLinearRegression(X, y)
    predictions = lm.predict(X)
    residuals = y - predictions
    assert_almost_equal(residuals.sum(), 0.)
    assert np.allclose(lm.theta, [2., 1.5])
