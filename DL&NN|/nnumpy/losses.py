"""
Error functions for training neural networks.

This module contains implementations for some common error functions.
"""

import numpy as np

from .base import Module
from .reductions import get_reduction

__all__ = ["LossFunction", "SquaredError", "LogisticError", "CrossEntropy",
           "LogitError", "LogitCrossEntropy"]


class LossFunction(Module):
    """ Base class for NNumpy loss functions. """

    def __init__(self, reduction='mean', target_grads=False):
        """
        Set up the loss function.

        Parameters
        ----------
        reduction : {'none', 'sum', 'mean'}, optional
            Specification of how to reduce the results on the sample dimension.
        target_grads : bool, optional
            Flag to enable gradients w.r.t. to the target values.
        """
        super().__init__()
        self.reduction = reduction
        self.disable_target_grads = not target_grads
        self.reduction = get_reduction(reduction, axis=0)

    def compute_outputs(self, predictions, targets):
        raw_out, cache = self.raw_outputs(predictions, targets)
        out, r_cache = self.reduction.compute_outputs(raw_out)
        return out, (cache, r_cache)

    def compute_grads(self, grads, cache):
        cache, r_cache = cache
        raw_grads = self.reduction.compute_grads(grads, r_cache)
        return self.raw_grads(raw_grads, cache)

    def raw_outputs(self, predictions, targets):
        raise NotImplementedError

    def raw_grads(self, grads, cache):
        raise NotImplementedError


class SquaredError(LossFunction):
    """ NNumpy implementation of the squared error loss function. """

    def raw_outputs(self, predictions, targets):
        diffs = predictions - targets.reshape(predictions.shape)
        return diffs ** 2 / 2, diffs

    def raw_grads(self, grads, diffs):
        dp = grads * diffs
        if self.disable_target_grads:
            return dp, np.nan

        return dp, -dp


class LogisticError(LossFunction):
    """ NNumpy implementation of the logistic error loss function. """

    def raw_outputs(self, predictions, targets):
        pos_entropy = -targets.reshape(predictions.shape) * np.log(predictions)
        neg_entropy = -(1 - targets) * np.log(1 - predictions)
        return pos_entropy + neg_entropy, (predictions, targets)

    def raw_grads(self, grads, cache):
        predictions, targets = cache
        dp = -targets / predictions + (1 - targets) / (1 - predictions)
        if self.disable_target_grads:
            return grads * dp, np.nan

        dt = -np.log(predictions) + np.log(1 - predictions)
        return grads * dp, grads * dt


class CrossEntropy(LossFunction):
    """ NNumpy implementation of the cross entropy loss function. """

    def raw_outputs(self, predictions, targets):
        entropy_terms = np.atleast_2d(targets) * np.log(predictions)
        entropy = -entropy_terms.sum(axis=-1, keepdims=True)
        return entropy, (predictions, targets)

    def raw_grads(self, grads, cache):
        predictions, targets = cache
        dp = -grads * targets / predictions
        if self.disable_target_grads:
            return dp, np.nan

        dt = -grads * np.log(predictions)
        return dp, dt


class LogitError(LossFunction):
    """
    NNumpy implementation of the logistic error loss function,
    computed from the logits, i.e. before applying the logistic sigmoid.
    """

    def raw_outputs(self, logits, targets):
        log_pred = -np.log(1 + np.exp(-logits))
        pos_entropy = -targets.reshape(logits.shape) * log_pred
        neg_entropy = -(1 - targets) * (log_pred - logits)
        return pos_entropy + neg_entropy, (log_pred, targets)

    def raw_grads(self, grads, cache):
        log_pred, targets = cache
        predictions = np.exp(log_pred)
        dp = grads * (predictions - targets)
        if self.disable_target_grads:
            return dp, np.nan

        dt = grads * (np.log(1 - predictions) - log_pred)
        return dp, dt


class LogitCrossEntropy(LossFunction):
    """
    NNumpy implementation of the cross entropy loss function
    computed from the logits, i.e. before applying the softmax nonlinearity.
    """

    def raw_outputs(self, logits, targets):
        logits = np.atleast_2d(logits)
        max_logit = np.max(logits, axis=-1, keepdims=True)
        sum_exp = np.sum(np.exp(logits - max_logit), axis=-1, keepdims=True)
        log_sum_exp = max_logit + np.log(sum_exp)
        log_softmax = logits - log_sum_exp
        entropy = -np.sum(targets * log_softmax, axis=-1, keepdims=True)
        return entropy, (log_softmax, targets)

    def raw_grads(self, grads, cache):
        log_softmax, targets = cache
        predictions = np.exp(log_softmax)
        dp = grads * (predictions - targets)
        if self.disable_target_grads:
            return dp, np.nan

        dt = grads * (-log_softmax)
        return dp, dt
