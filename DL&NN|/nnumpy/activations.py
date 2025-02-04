"""
Activation functions for neural networks.

This module contains implementations for
some commonly used activation functions.
"""

import numpy as np

from .base import Module

__all__ = [
    'Identity', 'LogisticSigmoid', 'SoftMax', 'Tanh',
    'AlgebraicSigmoid', 'Softplus', 'ReLU', 'ELU'
]


class Identity(Module):
    """ NNumpy implementation of the identity function. """
        
    def compute_outputs(self, s):
        return s, None
    
    def compute_grads(self, grads, cache):
        return grads


class LogisticSigmoid(Module):
    """ NNumpy implementation of the logistic sigmoid function. """
    
    def compute_outputs(self, s):
        a = 1 / (1 + np.exp(-s))
        return a, a
    
    def compute_grads(self, grads, cache):
        return grads * (1 - cache) * cache
    

class SoftMax(Module):
    """ NNumpy implementation of the softmax function. """
    
    def compute_outputs(self, s):
        # subtract max logit for numerical stability
        exp_s = np.exp(s - s.max())
        a = exp_s / np.sum(exp_s, axis=-1, keepdims=True)
        return a, a
        
    def compute_grads(self, grads, cache):
        # `grads * cache * (1 - cache)` does not quite do what we need it to do
        return cache * (grads - np.sum(grads * cache, axis=-1, keepdims=True))
    

class Tanh(Module):
    """ NNumpy implementation of the hyperbolic tangent function. """
        
    def compute_outputs(self, s):
        a = np.tanh(s)
        return a, a
    
    def compute_grads(self, grads, cache):
        return grads * (1 - cache ** 2)


class AlgebraicSigmoid(Module):
    """ NNumpy implementation of an algebraic sigmoid function. """

    def compute_outputs(self, s):
        cache = np.sqrt(1 + s ** 2)
        a = s / cache
        return a, cache

    def compute_grads(self, grads, cache):
        return grads / cache ** 3


class Softplus(Module):
    """ NNumpy implementation of the softplus function. """

    def compute_outputs(self, s):
        max_s = max(0, np.max(s))
        a = max_s + np.log(np.exp(-max_s) + np.exp(s - max_s))
        return a, s

    def compute_grads(self, grads, cache):
        return grads / (1 + np.exp(-cache))


class ReLU(Module):
    """ NNumpy implementation of the Rectified Linear Unit. """
    
    def compute_outputs(self, s):
        mask = s > 0
        return mask * s, mask
    
    def compute_grads(self, grads, cache):
        return grads * cache


class ELU(Module):
    """ NNumpy implementation of the Exponential Linear Unit. """

    def __init__(self, alpha=1.):
        super().__init__()
        if alpha < 0.:
            raise ValueError("negative values for alpha are not allowed")

        self.alpha = float(alpha)

    def compute_outputs(self, s):
        a = np.where(s > 0, s, self.alpha * (np.exp(s) - 1))
        return a, a

    def compute_grads(self, grads, cache):
        return grads * np.where(cache > 0, 1, cache + self.alpha)
