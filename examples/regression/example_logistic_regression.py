# -*- coding: utf-8 -*-

from __future__ import absolute_import

from packtml.regression import SimpleLogisticRegression
from packtml.utils.plotting import add_decision_boundary_to_axis
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
import sys

# #############################################################################
# Create an almost perfectly linearly-separable classification set
X, y = make_classification(n_samples=100, n_features=2, random_state=42,
                           n_redundant=0, n_repeated=0, n_classes=2,
                           class_sep=1.0)

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# #############################################################################
# Fit a simple logistic regression, produce predictions
lm = SimpleLogisticRegression(X_train, y_train, n_steps=50)

predictions = lm.predict(X_test)
acc = accuracy_score(y_test, predictions)
print("Test accuracy: %.3f" % acc)

# Show that our solution is similar to scikit-learn's
lr = LogisticRegression(fit_intercept=True, C=1e16)  # almost no regularization
lr.fit(X_train, y_train)
print("Sklearn test accuracy: %.3f" % accuracy_score(y_test,
                                                     lr.predict(X_test)))

# #############################################################################
# Plot the data and the boundary we learned.

add_decision_boundary_to_axis(estimator=lm, axis=plt,
                              nclasses=2, X_data=X_test)

# We have to break this into two plot calls, one for each class to
# have different markers...
c0_mask = y_test == 0
plt.scatter(X_test[c0_mask, 0], X_test[c0_mask, 1],
            c=~predictions[c0_mask], marker='o')
plt.scatter(X_test[~c0_mask, 0], X_test[~c0_mask, 1],
            c=~predictions[~c0_mask], marker='x')

plt.title("Logistic test performance: %.4f (o=true 0, x=true 1)" % acc)

# if we're supposed to save it, do so INSTEAD OF showing it
if len(sys.argv) > 1:
    plt.savefig(sys.argv[1])
else:
    plt.show()
