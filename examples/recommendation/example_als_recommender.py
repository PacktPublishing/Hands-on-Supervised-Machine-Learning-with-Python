# -*- coding: utf-8 -*-

from __future__ import absolute_import

from packtml.recommendation import ALS
from packtml.recommendation.data import get_completely_fabricated_ratings_data
from packtml.metrics.ranking import mean_average_precision
from matplotlib import pyplot as plt
import numpy as np
import sys

# #############################################################################
# Use our fabricated data set
R, titles = get_completely_fabricated_ratings_data()

# #############################################################################
# Fit an item-item recommender, predict for user 0
n_iter = 25
rec = ALS(R, factors=5, n_iter=n_iter, random_state=42, lam=0.01)
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

# plot the error
plt.plot(np.arange(n_iter), rec.train_err)
plt.xlabel("Iteration")
plt.ylabel("MSE")
plt.title("Train error by iteration")

# if we're supposed to save it, do so INSTEAD OF showing it
if len(sys.argv) > 1:
    plt.savefig(sys.argv[1])
else:
    plt.show()
