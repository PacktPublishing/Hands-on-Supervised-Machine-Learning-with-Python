# -*- coding: utf-8 -*-

from __future__ import absolute_import

from packtml.neural_net import NeuralNetClassifier
from sklearn.datasets import load_iris

iris = load_iris()
X, y = iris.data,  iris.target


def test_mlp():
    # show we can fit and predict
    clf = NeuralNetClassifier(X, y, random_state=42)
    clf.predict(X)
