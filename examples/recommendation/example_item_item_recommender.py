# -*- coding: utf-8 -*-

from __future__ import absolute_import

from packtml.recommendation import ItemItemRecommender
from packtml.recommendation.data import get_completely_fabricated_ratings_data
from packtml.metrics.ranking import mean_average_precision
import numpy as np

# #############################################################################
# Use our fabricated data set
R, titles = get_completely_fabricated_ratings_data()

# #############################################################################
# Fit an item-item recommender, predict for user 0
rec = ItemItemRecommender(R, k=3)
user0_rec, user_0_preds = rec.recommend_for_user(
    R, user=0, filter_previously_seen=True,
    return_scores=True)

# print some info about user 0
top_rated = np.argsort(-R[0, :])[:3]
print("User 0's top 3 rated movies are: %r" % titles[top_rated].tolist())
print("User 0's top 3 recommended movies are: %r"
      % titles[user0_rec[:3]].tolist())

# #############################################################################
# We can score our recommender as well, to determine how well it actually did

# first, get all user recommendations (top 10, not filtered)
recommendations = list(rec.recommend_for_all_users(
    R, n=10, filter_previously_seen=False,
    return_scores=False))

# get the TRUE items they've rated (in order)
ground_truth = np.argsort(-R, axis=1)
mean_avg_prec = mean_average_precision(
    predictions=recommendations, labels=ground_truth)
print("Mean average precision: %.3f" % mean_avg_prec)
