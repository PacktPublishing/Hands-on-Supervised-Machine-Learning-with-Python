# -*- coding: utf-8 -*-

from __future__ import absolute_import

from packtml.decision_tree.cart import RandomSplitter
from packtml.decision_tree.metrics import InformationGain
import numpy as np

# #############################################################################
# Build the example from the slides (3.3)
X = np.array([[21, 3], [ 4, 2], [37, 2]])
y = np.array([1, 0, 1])

# this is the splitting class; we'll use gini as the criteria
random_state = np.random.RandomState(42)
splitter = RandomSplitter(random_state=random_state,
                          criterion=InformationGain('gini'),
                          n_val_sample=3)

# find the best:
best_feature, best_value, best_gain = splitter.find_best(X, y)
print("Best feature=%i, best value=%r, information gain: %.3f"
      % (best_feature, best_value, best_gain))
