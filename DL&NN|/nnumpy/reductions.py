import numpy as np

from .activations import Identity
from .base import Module


__all__ = ["get_reduction", 'Sum', 'Mean', 'Max', 'Norm']


def get_reduction(name, axis=0, **kwargs):
    name = str(name).lower()
    if not name or name == 'none':
        return Identity()
    elif name == 'sum':
        return Sum(axis)
    elif name == 'max':
        return Max(axis)
    elif name == 'mean' or name == 'avg' or name == 'average':
        return Mean(axis)
    elif name == 'norm' or name == 'lp':
        return Norm(axis, **kwargs)
    else:
        raise ValueError("unknown aggregation: {}".format(name))


class Sum(Module):
    """
    NNumpy implementation of sum reduction.
    """

    def __init__(self, axis=0):
        super().__init__()
        self.axis = axis

    def compute_outputs(self, x):
        return np.sum(x, axis=self.axis), x.shape

    def compute_grads(self, grads, cache):
        grads = np.expand_dims(grads, self.axis)
        return np.broadcast_to(grads, cache)


class Mean(Module):
    """
    NNumpy implementation of mean reduction.
    """

    def __init__(self, axis=0):
        super().__init__()
        self.axis = axis

    def compute_outputs(self, x):
        return np.mean(x, axis=self.axis), x.shape

    def compute_grads(self, grads, cache):
        grads = np.expand_dims(grads, self.axis)
        return np.broadcast_to(grads, cache) / cache[self.axis]


class Max(Module):
    """
    NNumpy implementation of maximum reduction.
    """

    def __init__(self, axis=0):
        super().__init__()
        self.axis = axis

    def compute_outputs(self, x):
        arg_max = np.argmax(x, axis=self.axis)
        x_max = np.amax(x, axis=self.axis)
        return x_max, (arg_max, x.shape)

    def compute_grads(self, grads, cache):
        arg_max, shape = cache
        _grads = np.zeros(shape, grads.dtype)
        idx = np.ogrid[[slice(sh) for sh in grads.shape]]

        # append dummy element for correct behaviour with negative indices
        _idx = idx + [None]
        idx = _idx[:self.axis] + [arg_max] + _idx[self.axis:-1]

        _grads[tuple(idx)] = grads
        return _grads


class Norm(Module):
    """
    NNumpy implementation of Lp norm reduction.
    """

    def __init__(self, axis=0, p=2):
        super().__init__()
        if p <= 0:
            msg = "expected positive int for 'p', but was{}"
            raise ValueError(msg.format(p))

        self.p = int(p)
        self.axis = axis

    def compute_outputs(self, x):
        xp = np.abs(x) ** self.p
        sum_xp = np.sum(xp, axis=self.axis)
        out = sum_xp ** (1 / self.p)
        return out, (out, x)

    def compute_grads(self, grads, cache):
        out, x = cache
        out = np.expand_dims(out, self.axis)
        dx = out ** (1 - self.p) * np.abs(x) ** (self.p - 1) * np.sign(x)
        grads = np.expand_dims(grads, self.axis)
        return grads * dx
