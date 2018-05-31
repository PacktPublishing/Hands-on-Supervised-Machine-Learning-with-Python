# -*- coding: utf-8 -*-

from __future__ import absolute_import

from packtml.decision_tree.metrics import gini_impurity, InformationGain
import numpy as np

# #############################################################################
# Build the example from the slides
y = np.array([0, 0, 0, 1, 1, 1, 1])
uncertainty = gini_impurity(y)
print("Initial gini impurity: %.4f" % uncertainty)

# now get the information gain of the split from the slides
directions = np.array(["right", "left", "left", "left",
                       "right", "right", "right"])
mask = directions == "left"
print("Information gain from the split we created: %.4f"
      % InformationGain("gini")(target=y, mask=mask, uncertainty=uncertainty))
