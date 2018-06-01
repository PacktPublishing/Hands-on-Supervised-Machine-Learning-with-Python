# -*- coding: utf-8 -*-

from __future__ import absolute_import

from packtml.metrics.ranking import (mean_average_precision, ndcg_at,
                                     precision_at)

from numpy.testing import assert_almost_equal
import warnings

preds = [[1, 6, 2, 7, 8, 3, 9, 10, 4, 5],
         [4, 1, 5, 6, 2, 7, 3, 8, 9, 10],
         [1, 2, 3, 4, 5]]

labels = [[1, 2, 3, 4, 5], [1, 2, 3], []]


def assert_warning_caught(func):
    def test_wrapper(*args, **kwargs):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # execute the fxn
            func(*args, **kwargs)
            assert len(w)  # assert there's something there...
    return test_wrapper


@assert_warning_caught
def test_map():
    assert_almost_equal(
        mean_average_precision(preds, labels), 0.35502645502645497)


@assert_warning_caught
def test_pak():
    assert_almost_equal(precision_at(preds, labels, 1), 0.33333333333333331)
    assert_almost_equal(precision_at(preds, labels, 5), 0.26666666666666666)
    assert_almost_equal(precision_at(preds, labels, 15), 0.17777777777777778)


@assert_warning_caught
def test_ndcg():
    assert_almost_equal(ndcg_at(preds, labels, 3), 0.3333333432674408)
    assert_almost_equal(ndcg_at(preds, labels, 10), 0.48791273434956867)
