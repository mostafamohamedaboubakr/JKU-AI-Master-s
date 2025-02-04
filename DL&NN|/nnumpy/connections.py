"""
Connection models for neural networks.

This module contains implementations for different possibilities
to connect one layer of neurons in a neural network with the next layer.
"""

import numpy as np

from nnumpy.base import Module, Parameter
from nnumpy.utils import convolution_dot, as_tuple, symmetric_padding

__all__ = ['Linear', 'Convolution', 'Flatten']


class Linear(Module):
    """
    NNumpy implementation of a fully connected layer.

    Attributes
    ----------
    in_features : int
        Number of input features (D) this layer expects.
    out_features : int
        Number of output features (K) this layer produces.
    use_bias : bool
        Flag to indicate whether the bias parameters are used.

    w : Parameter
        Weight matrix.
    b : Parameter
        Bias vector.

    Examples
    --------
    >>> fc = Linear(10, 1)
    >>> fc.reset_parameters()  # init parameters
    >>> s = fc.forward(np.random.randn(1, 10))
    >>> fc.zero_grad()  # init gradients
    >>> ds = fc.backward(np.ones_like(s))
    """
    
    def __init__(self, in_features, out_features, use_bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        
        self.w = Parameter(shape=(out_features, in_features))
        if use_bias:
            self.b = Parameter(shape=out_features)
        self.reset_parameters()
        
    def reset_parameters(self, seed=None, w=None, b=0.):
        rng = np.random.default_rng(seed)
        if w is None:
            w = rng.standard_normal(self.w.shape)
        super().reset_parameters(seed=seed, w=w, b=b)
    
    def compute_outputs(self, x):
        s = x @ self.w.T
        if self.use_bias:
            s += self.b
        return s, x
    
    def compute_grads(self, grads, cache):
        if self.use_bias:
            self.b.grad = np.sum(grads, axis=0)

        self.w.grad = grads.T @ cache
        return grads @ self.w


class Convolution(Module):
    """
    NNumpy implementation of a ND convolutional layer.

    Attributes
    ----------
    in_channels : int
        Number of input channels this layer expects.
    out_channels : int
        Number of output channels this layer produces.
    kernel_size : tuple of ints
        Dimensions of each convolutional kernel.
    use_bias : bool
        Flag to indicate whether the bias parameters are used.

    w : Parameter
        Kernel tensor.
    b : Parameter
        Bias vector.

    Examples
    --------
    >>> conv = Convolution(10, 1, 5)
    >>> conv.reset_parameters()  # init parameters
    >>> s = conv.forward(np.random.randn(11, 1, 10))
    >>> conv.zero_grad()  # init gradients
    >>> ds = conv.backward(np.ones_like(s))
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 ndim=None, strides=1, use_bias=True):
        super().__init__()
        kernel_size = as_tuple(kernel_size, ndim)
        strides = as_tuple(strides, ndim or len(kernel_size))

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.use_bias = use_bias

        self.w = Parameter(shape=(out_channels, in_channels, *kernel_size))
        if use_bias:
            self.b = Parameter(shape=(out_channels, ) + (1, ) * self.ndim)
        self.reset_parameters()

    @property
    def ndim(self):
        return len(self.kernel_size)

    def reset_parameters(self, seed=None, w=None, b=0.):
        rng = np.random.default_rng(seed)
        if w is None:
            w = rng.standard_normal(self.w.shape)
        super().reset_parameters(seed=seed, w=w, b=b)

    def compute_outputs(self, x):
        corr = convolution_dot(self.w, x, stride=self.strides)
        if self.use_bias:
            corr += self.b
        return corr, x

    def compute_grads(self, grads, cache):
        if self.use_bias:
            b_grads = grads.reshape((len(grads), self.out_channels, -1))
            self.b.grad = np.sum(b_grads, axis=(0, -1)).reshape(self.b.shape)

        # TODO: resolve issues with non-aligned strides (shapes do not add up)
        dw_grads = np.array(grads, copy=False, ndmin=3)
        dw_x = np.array(cache, copy=False, ndmin=3)
        dw = convolution_dot(dw_grads.swapaxes(0, 1), dw_x.swapaxes(0, 1),
                             dilation=self.strides)
        self.w.grad = dw.swapaxes(0, 1)

        padding = np.asarray(self.kernel_size) - 1
        _grads = symmetric_padding(grads, padding, dilation=self.strides)
        reverse = (..., ) + (slice(None, None, -1), ) * self.ndim
        dx = convolution_dot(self.w[reverse].swapaxes(0, 1), _grads)
        return dx


class Flatten(Module):
    """
    NNumpy implementation of a flattening module.

    This kind of module is necessary to convert an image to a vector of pixels,
    as is necessary when going from a convolutional architecture
    to a fully connected one.
    """

    def compute_outputs(self, x):
        return x.reshape(len(x), -1), x.shape

    def compute_grads(self, grads, cache):
        return grads.reshape(cache)
