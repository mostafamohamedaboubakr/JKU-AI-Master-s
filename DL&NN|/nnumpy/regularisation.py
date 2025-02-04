import numpy as np

from .base import Module

__all__ = ["LpRegularisation", "Dropout"]


class LpRegularisation(Module):
    """ NNumpy implementation of Lp regularisation. """

    def __init__(self, p: int = 2):
        """
        Parameters
        ----------
        p : int
            The kind of norm to use for the regularisation.
        """
        super().__init__()
        self.p = p

    def backward(self, *grads):
        if self.predicting:
            raise ValueError("module not in training mode")

        self._shape_cache.pop()
        cache = self._forward_cache.pop()
        for grad in grads:
            self.compute_grads(grad, cache)

    def compute_outputs(self, parameters):
        cache = [(w, np.asarray(np.abs(w) ** self.p)) for w in parameters]

        total = 0.
        for _, exp in cache:
            total += np.sum(exp)

        return total / self.p, cache

    def compute_grads(self, grads, cache):
        for weight, exp in cache:
            weight.grad -= grads * exp / weight


class Dropout(Module):
    """ NNumpy implementation of (inverted) dropout. """

    def __init__(self, rate: float = .5, seed: int = None):
        """
        Parameters
        ----------
        rate : float, optional
            The percentage of neurons to be dropped.
        seed : int, optional
            Seed for the pseudo random generator.
        """
        super().__init__()
        if rate < 0. or rate > 1.:
            raise ValueError("dropout rate should be between zero and one")

        self.rate = float(rate)
        self.rng = np.random.default_rng(seed)

    def compute_outputs(self, x):
        if self.predicting or self.rate == 0.:
            return x, 1.

        keep_rate = 1. - self.rate
        mask = self.rng.binomial(1, keep_rate, size=x.shape)
        multiplier = mask / keep_rate
        return multiplier * x, multiplier

    def compute_grads(self, grads, cache):
        return grads * cache
