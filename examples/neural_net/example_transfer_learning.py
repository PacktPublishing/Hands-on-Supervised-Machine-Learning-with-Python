# -*- coding: utf-8 -*-

from __future__ import absolute_import

from packtml.neural_net import NeuralNetClassifier, TransferLearningClassifier
from packtml.utils.plotting import add_decision_boundary_to_axis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import sys

# #############################################################################
# Create a classification dataset. This dataset differs from other datsets
# we've created in that there are two majority classes, and one third (tiny)
# class that we'll train the transfer learner over
rs = np.random.RandomState(42)
covariance = [[1, .75], [.75, 1]]

# these are the majority classes
n_obs = 1250
x1 = rs.multivariate_normal(mean=[0, 0], cov=covariance, size=n_obs)
x2 = rs.multivariate_normal(mean=[1, 5], cov=covariance, size=n_obs)

# this is the minority class
x3 = rs.multivariate_normal(mean=[0.85, 3.25], cov=[[1., .5], [1.25, 0.85]],
                            size=n_obs // 3)

# this is what the FIRST network will be trained on
n_first = int(0.8 * n_obs)
X = np.vstack((x1[:n_first], x2[:n_first])).astype(np.float32)
y = np.hstack((np.zeros(n_first), np.ones(n_first))).astype(int)

# this is what the SECOND network will be trained on
X2 = np.vstack((x1[n_first:], x2[n_first:], x3)).astype(np.float32)
y2 = np.hstack((np.zeros(n_obs - n_first),
                np.ones(n_obs - n_first),
                np.ones(x3.shape[0]) * 2)).astype(int)

# split the data up
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=rs)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2,
                                                        random_state=rs)

# #############################################################################
# Fit the first neural network
hidden = (25, 25)
n_iter = 75
clf = NeuralNetClassifier(X_train, y_train, hidden=hidden, n_iter=n_iter,
                          learning_rate=0.001, random_state=42)

pred = clf.predict(X_test)
clf_accuracy = accuracy_score(y_test, pred)
print("Test accuracy (hidden=%s): %.3f" % (str(hidden), clf_accuracy))

# #############################################################################
# Fit the transfer network - train one more layer with a new class
t_hidden = (15,)
t_iter = 25
transfer = TransferLearningClassifier(X2_train, y2_train, pretrained=clf,
                                      hidden=t_hidden, n_iter=t_iter,
                                      random_state=42)

t_pred = transfer.predict(X2_test)
trans_accuracy = accuracy_score(y2_test, t_pred)
print("Test accuracy (hidden=%s): %.3f" % (str(hidden + t_hidden),
                                           trans_accuracy))

# #############################################################################
# Visualize how the models learned the classes

fig, axes = plt.subplots(2, 2, figsize=(12, 8))


add_decision_boundary_to_axis(estimator=clf, axis=axes[0, 0],
                              nclasses=2, X_data=X_test)
axes[0, 0].scatter(X_test[:, 0], X_test[:, 1], c=pred, alpha=0.4)
axes[0, 0].set_title("MLP network (hidden=%s @ %i iter): %.3f"
                     % (str(hidden), n_iter, clf_accuracy))

add_decision_boundary_to_axis(estimator=transfer, axis=axes[0, 1],
                              nclasses=3, X_data=X2_test)
axes[0, 1].scatter(X2_test[:, 0], X2_test[:, 1], c=t_pred, alpha=0.4)
axes[0, 1].set_title("Transfer network (hidden=%s @ %i iter): "
                     "%.3f" % (str(hidden + t_hidden), t_iter,
                               trans_accuracy))

# show the learning rates for each
axes[1, 0].plot(np.arange(len(clf.train_loss)), clf.train_loss)
axes[1, 0].set_title("Training loss by iteration")

# concat the two training losses together for this plot
trans_train_loss = clf.train_loss + transfer.train_loss
axes[1, 1].plot(np.arange(len(trans_train_loss)), trans_train_loss)
axes[1, 1].set_title("Training loss by iteration")

# Add a verticle line for where the transfer learning begins
axes[1, 1].axvline(x=n_iter, ls="--")

# if we're supposed to save it, do so INSTEAD OF showing it
if len(sys.argv) > 1:
    plt.savefig(sys.argv[1])
else:
    plt.show()
