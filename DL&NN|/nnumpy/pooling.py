import numpy as np

from .base import Module
from .reductions import get_reduction
from .utils import as_tuple, sig2col

__all__ = ["AvgPool2d", "MaxPool2d"]


class AvgPool2d(Module):
    """ Numpy DL implementation of an average-pooling layer. """

    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = tuple(kernel_size)

    def compute_outputs(self, x):
        """
        Parameters
        ----------
        x : (N, C, H, W) ndarray

        Returns
        -------
        a : (N, C, H', W') ndarray
        cache : ndarray or tuple of ndarrays
        """
        s = sig2col(x, self.kernel_size, stride=self.kernel_size)
        return np.mean(s, axis=(-1, -2)), np.array(x.shape)

    def compute_grads(self, grads, cache):
        """
        Parameters
        ----------
        grads : (N, C, H', W') ndarray
        cache : ndarray or tuple of ndarrays

        Returns
        -------
        dx : (N, C, H, W) ndarray
        """
        padding = np.asarray(self.kernel_size) - 1
        padded_shape = tuple(cache[:-2]) + tuple(cache[-2:] + padding)
        _grads = np.zeros(padded_shape, dtype=grads.dtype)
        idx = (...,) + tuple(slice(p, -p, k) for p, k in zip(padding, self.kernel_size))
        _grads[idx] = grads / np.prod(self.kernel_size)
        return sig2col(_grads, self.kernel_size).sum(axis=(-1, -2))


class MaxPool2d(Module):
    """ Numpy DL implementation of a max-pooling layer. """

    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = tuple(kernel_size)

    def compute_outputs(self, x):
        """
        Parameters
        ----------
        x : (N, C, H, W) ndarray

        Returns
        -------
        a : (N, C, H', W') ndarray
        cache : ndarray or tuple of ndarrays
        """
        s = sig2col(x, self.kernel_size, stride=self.kernel_size)
        s_flat = s.reshape(*s.shape[:-2], -1)
        indices = s_flat.argmax(axis=-1, keepdims=True)
        cache = (np.array(x.shape), np.array(s_flat.shape), indices)
        return np.take_along_axis(s_flat, indices, -1)[..., 0], cache

    def compute_grads(self, grads, cache):
        """
        Parameters
        ----------
        grads : (N, C, H', W') ndarray
        cache : ndarray or tuple of ndarrays

        Returns
        -------
        dx : (N, C, H, W) ndarray
        """
        x_shape, s_flat_shape, indices = cache
        s_grad_flat = np.zeros(s_flat_shape)
        np.put_along_axis(s_grad_flat, indices, grads[..., None], axis=-1)

        x_grad = np.zeros(x_shape)
        x_idx = np.arange(x_grad.size).reshape(x_shape)
        x_idx_s = sig2col(x_idx, self.kernel_size, stride=self.kernel_size)
        x_grad.flat[x_idx_s.ravel()] = s_grad_flat.ravel()
        return x_grad


class Pooling(Module):
    """ Numpy DL implementation of a general pooling layer. """

    def __init__(self, kernel_size, reduction='avg', ndim=None, strides=None):
        super().__init__()
        if strides is not None:
            raise NotImplementedError("strided pooling not yet implemented")

        self.kernel_size = as_tuple(kernel_size, ndim)
        ndim = ndim or len(self.kernel_size)
        self.strides = as_tuple(strides or kernel_size, ndim)
        self.reduction = get_reduction(reduction, axis=-1)

    @property
    def ndim(self):
        return len(self.kernel_size)

    def compute_outputs(self, x):
        cols = sig2col(x, self.kernel_size, stride=np.array(self.strides))
        cols = cols.reshape(*cols.shape[:-self.ndim], -1)
        return self.reduction.compute_outputs(cols)

    def compute_grads(self, grads, cache):
        ds = self.reduction.compute_grads(grads, cache)
        ds = ds.reshape(ds.shape[:-1] + self.kernel_size)
        r = np.zeros(cache)

        s = self.strides
        d = self.dilation
        for i, dsi in enumerate(ds):
            r[s * i:s * i + d * self.kernel_size[0]:d] += dsi
        print(ds.shape)
        # TODO: backprop through sig2col
        return np.zeros_like(cache)
