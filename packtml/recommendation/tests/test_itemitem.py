# -*- coding: utf-8 -*-

from __future__ import absolute_import

from packtml.recommendation import ItemItemRecommender

import numpy as np
from numpy.testing import assert_array_almost_equal

from types import GeneratorType

# make up a ratings matrix...
R = np.array([[1.,  0.,  3.5,  2.,  0.,  0.,  0.,  1.5],
              [0.,  2.,  3.,   0.,  0.,  2.5, 0.,  0. ],
              [3.5, 4.,  2.,   0.,  4.5, 3.5, 0.,  2. ],
              [3.,  3.5, 0.,   2.5, 3.,  0.,  0.,  0. ]])


def test_itemitem_simple():
    rec = ItemItemRecommender(R, k=3)

    # assert on the similarity
    expected = np.array([
        [ 1.        ,  0.91461057,  0.        ,  0.        ,  0.9701687 ,
          0.        ,  0.        ,  0.        ],
        [ 0.91461057,  1.        ,  0.        ,  0.        ,  0.92793395,
          0.        ,  0.        ,  0.        ],
        [ 0.        ,  0.        ,  1.        ,  0.        ,  0.        ,
          0.6708902 ,  0.        ,  0.73632752],
        [ 0.62906665,  0.48126166,  0.        ,  1.        ,  0.        ,
          0.        ,  0.        ,  0.        ],
        [ 0.9701687 ,  0.92793395,  0.        ,  0.        ,  1.        ,
          0.        ,  0.        ,  0.        ],
        [ 0.        ,  0.77786258,  0.        ,  0.        ,  0.67706717,
          1.        ,  0.        ,  0.        ],
        [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
          0.        ,  0.        ,  0.        ],
        [ 0.72079856,  0.        ,  0.73632752,  0.        ,  0.        ,
          0.        ,  0.        ,  1.        ]])

    assert_array_almost_equal(expected, rec.similarity)

    # show we can generate recommendations
    rec0, scores0 = rec.recommend_for_user(R, 0)

    # we didn't filter, so the rated items should still be present
    assert np.in1d([0, 2, 3, 7], rec0).all()

    # re-compute and show the previously-rated are not present
    rec0_filtered, scores0_filtered = rec.recommend_for_user(
        R, 0, filter_previously_seen=True)

    assert len(rec0_filtered) == 4, rec0_filtered
    assert rec0_filtered.tolist() == [5, 1, 4, 6]

    # test the prediction, which is just a big product...
    pred = rec.predict(R)
    assert pred.shape == R.shape

    # get recommendations for ALL users
    recommendations = rec.recommend_for_all_users(R, return_scores=False,
                                                  filter_previously_seen=False)

    assert isinstance(recommendations, GeneratorType)
    recs = list(recommendations)
    assert len(recs) == 4
    assert all(len(x) == 8 for x in recs)
