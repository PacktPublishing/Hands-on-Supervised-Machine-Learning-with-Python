# -*- coding: utf-8 -*-

from __future__ import absolute_import

from packtml.decision_tree import CARTClassifier
from packtml.utils.plotting import add_decision_boundary_to_axis
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import sys

# #############################################################################
# Create a classification dataset
rs = np.random.RandomState(42)
covariance = [[1, .75], [.75, 1]]
n_obs = 500
x1 = rs.multivariate_normal(mean=[0, 0], cov=covariance, size=n_obs)
x2 = rs.multivariate_normal(mean=[1, 3], cov=covariance, size=n_obs)

X = np.vstack((x1, x2)).astype(np.float32)
y = np.hstack((np.zeros(n_obs), np.ones(n_obs)))

# split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# #############################################################################
# Fit a simple decision tree classifier and get predictions
shallow_depth = 2
clf = CARTClassifier(X_train, y_train, max_depth=shallow_depth, criterion='gini',
                     random_state=42)
pred = clf.predict(X_test)
clf_accuracy = accuracy_score(y_test, pred)
print("Test accuracy (depth=%i): %.3f" % (shallow_depth, clf_accuracy))

# Fit a deeper tree and show accuracy increases
clf2 = CARTClassifier(X_train, y_train, max_depth=25, criterion='gini',
                      random_state=42)
pred2 = clf2.predict(X_test)
clf2_accuracy = accuracy_score(y_test, pred2)
print("Test accuracy (depth=25): %.3f" % clf2_accuracy)

# #############################################################################
# Visualize difference in classification ability

fig, axes = plt.subplots(1, 2, figsize=(12, 8))

add_decision_boundary_to_axis(estimator=clf, axis=axes[0],
                              nclasses=2, X_data=X_test)
axes[0].scatter(X_test[:, 0], X_test[:, 1], c=pred, alpha=0.4)
axes[0].set_title("Shallow tree (depth=%i) performance: %.3f"
                  % (shallow_depth, clf_accuracy))

add_decision_boundary_to_axis(estimator=clf2, axis=axes[1],
                              nclasses=2, X_data=X_test)
axes[1].scatter(X_test[:, 0], X_test[:, 1], c=pred2, alpha=0.4)
axes[1].set_title("Deep tree (depth=25) performance: %.3f" % clf2_accuracy)

# if we're supposed to save it, do so INSTEAD OF showing it
if len(sys.argv) > 1:
    plt.savefig(sys.argv[1])
else:
    plt.show()
