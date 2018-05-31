# -*- coding: utf-8 -*-

from __future__ import absolute_import

from packtml.regression import SimpleLogisticRegression
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

import numpy as np

X, y = make_classification(n_samples=100, n_features=2, random_state=42,
                           n_redundant=0, n_repeated=0, n_classes=2,
                           class_sep=1.0)


def test_simple_logistic():
    lm = SimpleLogisticRegression(X, y, n_steps=50, loglik_interval=10)
    assert np.allclose(lm.theta, np.array([ 1.32320936, -0.03926072]))

    # test that we can predict
    preds = lm.predict(X)

    # show we're better than chance
    assert accuracy_score(y, preds) > 0.5

    # show that we only computed the log likelihood 5 times
    assert len(lm.log_likelihood) == 5, lm.log_likelihood
