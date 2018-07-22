# -*- coding: utf-8 -*-

from __future__ import absolute_import

from packtml.clustering import KNNClassifier
from packtml.utils.plotting import add_decision_boundary_to_axis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import sys

# #############################################################################
# Create a classification sub-dataset using iris
iris = load_iris()
X = iris.data[:, :2]  # just use the first two dimensions
y = iris.target

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# #############################################################################
# Fit a k-nearest neighbor model and get predictions
k=10
clf = KNNClassifier(X_train, y_train, k=k)
pred = clf.predict(X_test)
clf_accuracy = accuracy_score(y_test, pred)
print("Test accuracy: %.3f" % clf_accuracy)

# #############################################################################
# Visualize difference in classes (this is from the scikit-learn KNN
# plotting example:
# http://scikit-learn.org/stable/auto_examples/neighbors/plot_classification.html#sphx-glr-auto-examples-neighbors-plot-classification-py)

xx, yy, _ = add_decision_boundary_to_axis(estimator=clf, axis=plt,
                                          nclasses=3, X_data=X_test)

# Plot also the training points
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test,
            cmap=ListedColormap(['#FF0000', '#00FF00', '#0000FF']),
            edgecolor='k', s=20)

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("3-Class classification (k=%i)" % k)

# if we're supposed to save it, do so INSTEAD OF showing it
if len(sys.argv) > 1:
    plt.savefig(sys.argv[1])
else:
    plt.show()
