# -*- coding: utf-8 -*-

from __future__ import absolute_import

from packtml.recommendation import ALS

# make up a ratings matrix...
R = [[1.,  0.,  3.5,  2.,  0.,  0.,  0.,  1.5],
     [0.,  2.,  3.,   0.,  0.,  2.5, 0.,  0. ],
     [3.5, 4.,  2.,   0.,  4.5, 3.5, 0.,  2. ],
     [3.,  3.5, 0.,   2.5, 3.,  0.,  0.,  0. ]]


def test_als_simple_fit():
    als = ALS(R, factors=3, n_iter=5, random_state=42)
    assert len(als.train_err) == 5, als.train_err
    assert als.n_factors == 3, als.n_factors

    # assert all errors are decreasing over time
    errs = list(zip(als.train_err[:-1], als.train_err[1:]))
    assert all(new_err < last_err for last_err, new_err in errs), errs


def test_als_predict():
    als = ALS(R, factors=4, n_iter=8, random_state=42)
    user0, scr = als.recommend_for_user(R, 0, filter_previously_seen=True,
                                        return_scores=True)

    # assert previously-rated items not present
    rated = (0, 2, 3, 7)
    for r in rated:  # previously-rated
        assert r not in user0

    # show the score lengths are the same
    assert scr.shape[0] == user0.shape[0]

    # now if we do NOT filter, assert those are present again (also, recompute)
    user0, scr = als.recommend_for_user(R, 0, filter_previously_seen=False,
                                        return_scores=True,
                                        recompute_user=True)
    for r in rated:
        assert r in user0

    assert user0.shape[0] == scr.shape[0]
