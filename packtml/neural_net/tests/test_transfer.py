# -*- coding: utf-8 -*-

from __future__ import absolute_import

from packtml.neural_net import NeuralNetClassifier, TransferLearningClassifier

import numpy as np


def test_transfer_learner():
    rs = np.random.RandomState(42)
    covariance = [[1, .75], [.75, 1]]

    # these are the majority classes
    n_obs = 500
    x1 = rs.multivariate_normal(mean=[0, 0], cov=covariance, size=n_obs)
    x2 = rs.multivariate_normal(mean=[1, 5], cov=covariance, size=n_obs)

    # this is the minority class
    x3 = rs.multivariate_normal(mean=[0.85, 3.25],
                                cov=[[1., .5], [1.25, 0.85]],
                                size=150)

    # this is what the FIRST network will be trained on
    n_first = 400
    X = np.vstack((x1[:n_first], x2[:n_first])).astype(np.float32)
    y = np.hstack((np.zeros(n_first), np.ones(n_first))).astype(int)

    # this is what the SECOND network will be trained on
    X2 = np.vstack((x1[n_first:], x2[n_first:], x3)).astype(np.float32)
    y2 = np.hstack((np.zeros(n_obs - n_first),
                    np.ones(n_obs - n_first),
                    np.ones(x3.shape[0]) * 2)).astype(int)

    # Fit the first neural network
    clf = NeuralNetClassifier(X, y, hidden=(25, 25), n_iter=50,
                              learning_rate=0.001, random_state=42)

    # Fit the transfer network - train one more layer with a new class
    transfer = TransferLearningClassifier(X2, y2, pretrained=clf, hidden=(15,),
                                          n_iter=10, random_state=42)

    # show we can predict
    transfer.predict(X2)

    # show we can use a transfer learner on an existing transfer learner
    transfer2 = TransferLearningClassifier(X2, y2, pretrained=transfer,
                                           hidden=(25,),
                                           random_state=15)

    # and show we can still predict
    transfer2.predict(X2)
