# -*- coding: utf-8 -*-

from __future__ import absolute_import

from packtml.neural_net import NeuralNetClassifier
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
n_obs = 1000
x1 = rs.multivariate_normal(mean=[0, 0], cov=covariance, size=n_obs)
x2 = rs.multivariate_normal(mean=[1, 5], cov=covariance, size=n_obs)

X = np.vstack((x1, x2)).astype(np.float32)
y = np.hstack((np.zeros(n_obs), np.ones(n_obs))).astype(int)

# split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=rs)

# #############################################################################
# Fit a simple neural network
n_iter = 4
hidden = (10,)
clf = NeuralNetClassifier(X_train, y_train, hidden=hidden, n_iter=n_iter,
                          learning_rate=0.001, random_state=42)
print("Loss per training iteration: %r" % clf.train_loss)

pred = clf.predict(X_test)
clf_accuracy = accuracy_score(y_test, pred)
print("Test accuracy (hidden=%s): %.3f" % (str(hidden), clf_accuracy))

# #############################################################################
# Fit a more complex neural network
n_iter2 = 150
hidden2 = (25, 25)
clf2 = NeuralNetClassifier(X_train, y_train, hidden=hidden2, n_iter=n_iter2,
                           learning_rate=0.001, random_state=42)

pred2 = clf2.predict(X_test)
clf_accuracy2 = accuracy_score(y_test, pred2)
print("Test accuracy (hidden=%s): %.3f" % (str(hidden2), clf_accuracy2))

# #############################################################################
# Visualize difference in classification ability

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

add_decision_boundary_to_axis(estimator=clf, axis=axes[0, 0],
                              nclasses=2, X_data=X_test)
axes[0, 0].scatter(X_test[:, 0], X_test[:, 1], c=pred, alpha=0.4)
axes[0, 0].set_title("Shallow (hidden=%s @ %i iter) test accuracy: %.3f"
                     % (str(hidden), n_iter, clf_accuracy))

add_decision_boundary_to_axis(estimator=clf2, axis=axes[0, 1],
                              nclasses=2, X_data=X_test)
axes[0, 1].scatter(X_test[:, 0], X_test[:, 1], c=pred2, alpha=0.4)
axes[0, 1].set_title("Deeper (hidden=%s @ %i iter): test accuracy: %.3f"
                     % (str(hidden2), n_iter2, clf_accuracy2))

# show the learning rates for each
axes[1, 0].plot(np.arange(len(clf.train_loss)), clf.train_loss)
axes[1, 0].set_title("Training loss by iteration")

axes[1, 1].plot(np.arange(len(clf2.train_loss)), clf2.train_loss)
axes[1, 1].set_title("Training loss by iteration")

# if we're supposed to save it, do so INSTEAD OF showing it
if len(sys.argv) > 1:
    plt.savefig(sys.argv[1])
else:
    plt.show()
