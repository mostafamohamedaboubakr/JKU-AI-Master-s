import numpy as np

from .base import Module
from .containers import Container
from . import init

__all__ = ["BatchNormalisation", "LayerNormalisation", "WeightNormalisation"]


class BatchNormalisation(Module):
    """ NNumpy implementation of batch normalisation. """

    def __init__(self, dims: tuple, normalised: bool = False, eps: float = 1e-8):
        """
        Parameters
        ----------
        dims : tuple of ints
            The shape of the incoming signal (without batch dimension).
        normalised : bool, optional
            Flag to disable the gamma and beta parameters.
        eps : float, optional
            Small value for numerical stability.
        """
        super().__init__()
        self.dims = tuple(dims)
        self.normalised = normalised
        self.eps = float(eps)

        if not normalised:
            self.gamma = self.register_parameter('gamma', np.ones(self.dims))
            self.beta = self.register_parameter('beta', np.zeros(self.dims))

        self.running_count = 0
        self.running_sums = np.zeros((2,) + self.dims)

    def compute_outputs(self, x):
        if self.predicting:
            batch_mean, batch_var = self.running_sums / self.running_count
        else:
            # safety check, I do not expect students to implement this
            x = x.reshape(-1, *self.dims)
            if len(x) <= 1:
                raise ValueError("can not do batch norm with single sample")

            # compute batch mean and var
            batch_mean = np.mean(x, axis=0)
            batch_var = np.mean((x - batch_mean) ** 2, axis=0)
            self.running_sums += np.stack([batch_mean, batch_var])

            # cumulative averaging
            self.running_count += 1

        batch_std = np.sqrt(batch_var) + self.eps
        result = (x - batch_mean) / batch_std
        cache = (result, batch_std)
        if not self.normalised:
            result = self.gamma * result + self.beta

        return result, cache

    def compute_grads(self, grads, cache):
        normalised, std = cache
        if not self.normalised:
            self.gamma.grad = np.sum(grads * normalised, axis=0)
            self.beta.grad = np.sum(grads, axis=0)
            grads = grads * self.gamma

        d1 = grads / std
        d2 = -normalised * np.mean(normalised * d1, axis=0)
        d3 = -np.mean(d1, axis=0)

        return d1 + d2 + d3


class LayerNormalisation(Module):

    def __init__(self, dims: tuple, eps: float = 1e-8):
        super().__init__()
        self.dims = tuple(dims)
        self.eps = float(eps)

    def compute_outputs(self, x):
        x = x.reshape(*x.shape[:-len(self.dims)], -1)
        act_mean = np.mean(x, axis=-1, keepdims=True)
        act_var = np.mean((x - act_mean) ** 2, axis=-1, keepdims=True)
        act_std = np.sqrt(act_var + self.eps)
        result = (x - act_mean) / act_std
        result = result.reshape(*x.shape[:-1], *self.dims)
        cache = (result, act_std)
        return result, cache

    def compute_grads(self, grads, cache):
        normalised, std = cache
        d1 = grads / std
        d2 = -normalised * np.mean(normalised * d1, axis=-1, keepdims=True)
        d3 = -np.mean(d1, axis=-1, keepdims=True)
        return d1 + d2 + d3


class WeightNormalisation(Container):
    """ NNumpy implementation of weight normalisation. """

    def __init__(self, module):
        super().__init__()
        self.connection = self.add_module(module, 'connection')
        self.g = self.register_parameter('g', np.empty(1))

    def reset_parameters(self):
        init.ones(self.g)

    def compute_outputs(self, x):
        s, cache = self.connection.compute_outputs(x)
        weight_norm = np.linalg.norm(self.connection.w)
        scale = self.g / weight_norm
        return scale * s, (s, weight_norm, cache)

    def compute_grads(self, grads, cache):
        x, weight_norm, conn_cache = cache
        self.g.grad = dg = np.sum(grads * x) / weight_norm

        scale1 = self.g / weight_norm
        dx = self.connection.compute_grads(scale1 * grads, conn_cache)
        scale2 = scale1 * dg / weight_norm
        self.connection.w.grad -= scale2 * self.connection.w

        return dx
