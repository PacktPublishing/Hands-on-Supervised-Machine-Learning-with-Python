# -*- coding: utf-8 -*-
#
# Author: Taylor G Smith <taylor.smith@alkaline-ml.com>
#
# A simple transfer learning classifier. If you find yourself struggling
# to follow the derivation of the back-propagation, check out this great
# refresher on scalar & matrix calculas + differential equations.
# http://parrt.cs.usfca.edu/doc/matrix-calculus/index.html

from __future__ import absolute_import

import numpy as np

from packtml.neural_net.base import NeuralMixin, tanh
from packtml.base import BaseSimpleEstimator
from packtml.neural_net.mlp import NeuralNetClassifier, _calculate_loss

__all__ = [
    'TransferLearningClassifier'
]

try:
    xrange
except NameError:
    xrange = range


def _pretrained_forward_step(X, pt_weights, pt_biases):
    """Complete a forward step from the pre-trained model"""
    # progress through all the layers (the output was already trimmed off)
    for w, b in zip(pt_weights, pt_biases):
        X = tanh(X.dot(w) + b)
    return X


class TransferLearningClassifier(BaseSimpleEstimator, NeuralMixin):
    """A transfer learning classifier.

    Create a multi-layer perceptron classifier that learned from a
    previously-trained network. No fine-tuning is performed, and no
    prior-trained layers can be retrained (i.e., they remain frozen).

    Parameters
    ----------
    X : array-like, shape=(n_samples, n_features)
        The training array. Should be a numpy array or array-like structure
        with only finite values.

    y : array-like, shape=(n_samples,)
        The target vector.

    pretrained : NeuralNetClassifier, TransferLearningClassifier
        The pre-trained MLP. The transfer learner leverages the features
        extracted from the pre-trained network (the trained weights without
        the output layer) and uses them to transform the input data before
        training the new layers.

    hidden : iterable, optional (default=(25,))
        An iterable indicating the number of units per hidden layer.

    n_iter : int, optional (default=10)
        The default number of iterations to perform.

    learning_rate : float, optional (default=0.001)
        The rate at which we descend the gradient.

    random_state : int, None or RandomState, optional (default=42)
        The random state for initializing the weights matrices.
    """
    def __init__(self, X, y, pretrained, hidden=(25,), n_iter=10,
                 regularization=0.01, learning_rate=0.001, random_state=42):

        # initialize via the NN static method
        self.hidden = hidden
        self.random_state = random_state
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.regularization = regularization

        # this is the previous model
        self.model = pretrained

        # assert that it's a neural net or we'll break down later
        assert isinstance(pretrained, NeuralMixin), \
            "Pre-trained model must be a neural network!"

        # initialize weights, biases, etc. for THE TRAINABLE LAYERS ONLY!
        pt_w, pt_b = pretrained.export_weights_and_biases(output_layer=False)
        X, y, weights, biases = NeuralNetClassifier._init_weights_biases(
            X, y, hidden, random_state,

            # use as the last dim the column dimension of the last weights
            # (the ones BEFORE the output layer, that is)
            last_dim=pt_w[-1].shape[1])

        # we can train this in a similar fashion to the plain MLP we designed:
        # for each iteration, feed X through the network, compute the loss,
        # and back-propagate the error to correct the weights.
        train_loss = []
        for _ in xrange(n_iter):
            # first, pass the input data through the pre-trained model's
            # hidden layers. Do not pass it through the last layer, however,
            # since we don't want its output from the softmax layer.
            X_transform = _pretrained_forward_step(X, pt_w, pt_b)

            # NOW we complete a forward step on THIS model's
            # untrained  weights/biases
            out, layer_results = NeuralNetClassifier._forward_step(
                X_transform, weights, biases)

            # compute the loss on the output
            loss = _calculate_loss(truth=y, preds=out, weights=pt_w + weights,
                                   l2=self.regularization)
            train_loss.append(loss)

            # now back-propagate to correct THIS MODEL's weights and biases via
            # gradient descent. NOTE we do NOT adjust the pre-trained model's
            # weights!!!
            NeuralNetClassifier._back_propagate(
                truth=y, probas=out, layer_results=layer_results,
                weights=weights, biases=biases,
                learning_rate=learning_rate,
                l2=self.regularization)

        # save the weights, biases
        self.weights = weights
        self.biases = biases
        self.train_loss = train_loss

    def predict(self, X):
        # compute the probabilities and then get the argmax for each class
        probas = self.predict_proba(X)

        # we want the argmaxes of each row
        return np.argmax(probas, axis=1)

    def predict_proba(self, X):
        # Compute a forward step with the pre-trained model first:
        pt_w, pt_b = self.model.export_weights_and_biases(output_layer=False)
        X_transform = _pretrained_forward_step(X, pt_w, pt_b)

        # and then complete a forward step with the trained weights and biases
        return NeuralNetClassifier._forward_step(
            X_transform, self.weights, self.biases)[0]

    def export_weights_and_biases(self, output_layer=True):
        pt_weights, pt_biases = \
            self.model.export_weights_and_biases(output_layer=False)
        w = pt_weights + self.weights
        b = pt_biases + self.biases

        if output_layer:
            return w, b
        return w[:-1], b[:-1]
