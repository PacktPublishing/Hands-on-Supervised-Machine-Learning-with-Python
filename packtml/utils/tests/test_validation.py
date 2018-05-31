# -*- coding: utf-8 -*-

from __future__ import absolute_import

from packtml.utils import validation as val
from packtml.regression import SimpleLogisticRegression

from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer

bc = load_breast_cancer()
X, y = bc.data, bc.target


def test_is_iterable():
    assert val.is_iterable([1, 2, 3])
    assert val.is_iterable((1, 2, 3))
    assert val.is_iterable({1, 2, 3})
    assert val.is_iterable({1: 'a', 2: 'b'})
    assert not val.is_iterable(123)
    assert not val.is_iterable(None)
    assert not val.is_iterable("a string")


def test_learning_curves():
    train_scores, val_scores = \
        val.learning_curve(
            SimpleLogisticRegression, X, y,
            metric=accuracy_score,
            train_sizes=(100, 250, 400),
            n_folds=3, seed=42, trace=True,

            # kwargs:
            n_steps=20, loglik_interval=20)

    assert train_scores.shape == (3, 3)
    assert val_scores.shape == (3, 3)
