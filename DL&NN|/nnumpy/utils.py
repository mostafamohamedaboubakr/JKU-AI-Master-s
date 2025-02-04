import numpy as np


def as_tuple(x, length=None):
    """
    Utility function for converting anything into a tuple.

    Parameters
    ----------
    x : object
        The object to convert to a tuple.
    length : int, optional
        The required length for the tuple.

    Raises
    ------
    ValueError : if iterable does not have the correct length.

    Returns
    -------
    res : tuple
        The result of calling `tuple(x)` if possible,
        else a tuple with `length` times `x`.
    """
    try:
        _x = tuple(x)
        if length is not None and len(_x) != length:
            msg = "expected argument with length {:d}, but was {:d}"
            raise ValueError(msg.format(length, len(_x)))
    except TypeError:
        _x = (x, )
        if length is not None:
            _x = _x * length

    return _x


def split_data(x, y, ratio=3):
    """
    Split up a dataset in two subsets with a specific ratio.

    Parameters
    ----------
    x : ndarray
        Input features.
    y : ndarray
        Target values.
    ratio : int or float, optional
        If an integer is provided,
        the first split will have `ratio` as many samples as the second split.
        A value between 0.5 and 1 corresponds to
        the percentage of data that goes to the first split.

    Returns
    -------
    (x1, y1) : tuple of ndarrays
        The first, larger split of the dataset.
    (x2, y2) : tuple of ndarrays
        The second, smaller split of the dataset.
    """
    # not the most efficient solution, but elegant
    step = int(1 / (1 - ratio)) if ratio < 1 else int(ratio) + 1
    idx = slice(None, None, step)
    x1, x2 = np.delete(x, idx, axis=0), x[idx]
    y1, y2 = np.delete(y, idx, axis=0), y[idx]
    return (x1, y1), (x2, y2)


def split_shuffled_data(x, y, ratio=3, seed=None):
    """
    Shuffle and split up a dataset in two subsets with a specific ratio.

    Parameters
    ----------
    x : ndarray
        Input features.
    y : ndarray
        Target values.
    ratio : int or float, optional
        If an integer is provided,
        the first split will have `ratio` as many sample as the second split.
        A value between 0.5 and 1 corresponds to
        the percentage of data that goes to the first split.
    seed : int, optional
        Value that allows to control the randomness.

    Returns
    -------
    (x1, y1) : tuple of ndarrays
        The first split of the dataset.
    (x2, y2) : tuple of ndarrays
        The second split of the dataset.
    """
    if ratio > 1:
        ratio = 1 - 1 / (1 + ratio)

    rng = np.random.default_rng(seed)
    rng_state = rng.bit_generator.state
    rng.shuffle(x)
    rng.bit_generator.state = rng_state
    rng.shuffle(y)

    split = int(ratio * len(x))
    x1, x2 = np.split(x, [split], axis=0)
    y1, y2 = np.split(y, [split], axis=0)
    return (x1, y1), (x2, y2)


def to_one_hot(y, k=None):
    """
    Compute a one-hot encoding from a vector of integer labels.

    Parameters
    ----------
    y : (N1, ..., Nd) ndarray
        The zero-indexed integer labels to encode.
    k : int, optional
        The number of distinct labels in `y`.

    Returns
    -------
    one_hot : (N1, ..., Nd, k) ndarray
        The one-hot encoding of the labels.
    """
    y = np.asarray(y, dtype='int')
    if k is None:
        k = np.amax(y) + 1

    out = np.zeros(y.shape + (k, ))
    np.put_along_axis(out, y[..., None], 1, axis=-1)
    return out


def symmetric_padding(x, pad=0, dilation=1, ndim=None):
    """
    Zero-pad numpy arrays symmetrically with option for dilation padding.

    Parameters
    ----------
    x : ndarray
        Numpy array to be padded.
    pad : int or tuple, optional
        Padding on the side along each axis (starting from the back).
    dilation : int or tuple, optional
        Padding inbetween the data along each axis (starting from the back).
    ndim : int, optional
        Number of axis to pad (starting from the back).

    Returns
    -------
    padded : ndarray
        The zero-padded numpy array.
    """
    x = np.asarray(x)
    pad = as_tuple(pad, ndim)
    ndim = ndim or len(pad)
    dilation = np.asarray(dilation)

    x_shape1, x_shape2 = np.split(x.shape, [-ndim])
    _shape = dilation * (x_shape2 - 1) + 1
    dilated = np.zeros(tuple(x_shape1) + tuple(_shape), x.dtype)
    idx = tuple(slice(None, None, d) for d in as_tuple(dilation, ndim))
    dilated[(..., ) + idx] = x

    pads = (0, ) * len(x_shape1) + pad
    return np.pad(dilated, tuple(zip(pads, pads)))


def multi_channel_convolution1d(w, x, stride=1, dilation=1):
    """
    Implementation of the 1D convolution operation
    with multiple in- and output channels.

    Parameters
    ----------
    w : (Co, Ci, R) ndarray
        Weights for the 1D convolution.
    x : (N, Ci, L) ndarray
        Input data for the 1D convolution.
    stride : int, optional
        Stride for the 1D convolution.
    dilation : int, optional
        Dilation for the 1D convolution.

    Returns
    -------
    corr : (N, Co, L') ndarray
        1D cross-correlation of `x` and `w`.
    """
    w = np.array(w, copy=False, ndmin=3)
    x = np.array(x, copy=False, ndmin=2)
    c_out, c_in, r = w.shape

    if c_in != x.shape[-2]:
        msg = "input channel dimensions do not match: {:d} != {:d}"
        raise ValueError(msg.format(c_in, x.shape[-2]))

    kernel_size = dilation * (r - 1) + 1
    l_out = (x.shape[-1] - kernel_size) // stride + 1
    res = np.zeros(x.shape[:-2] + (c_out, l_out), x.dtype)
    for i, wi in enumerate(w.T):
        i_dilated = i * dilation
        x_part = x[..., i_dilated:i_dilated + l_out * stride:stride]
        res += np.einsum('...ij,ik->...kj', x_part, wi, optimize=True)
    return res


def multi_channel_convolution2d(w, x, stride=(1, 1), dilation=(1, 1)):
    """
    Implementation of the 2D convolution operation
    with multiple in- and output channels.

    Parameters
    ----------
    w : (Co, Ci, R1, R2) ndarray
        Weights for the 2D convolution.
    x : (N, Ci, H, W) ndarray
        Input data for the 2D convolution.
    stride : tuple of ints, optional
        2D strides for the convolution.
    dilation : tuple of ints, optional
        2D dilations for the convolution.

    Returns
    -------
    corr : (N, Co, H', W') ndarray
        2D cross-correlation of `x` and `w`.
    """
    w = np.array(w, copy=False, ndmin=4)
    x = np.array(x, copy=False, ndmin=3)
    c_out, c_in, r1, r2 = w.shape

    if c_in != x.shape[-3]:
        msg = "input channel dimensions do not match: {:d} != {:d}"
        raise ValueError(msg.format(c_in, x.shape[-3]))

    stride1, stride2 = stride
    dil1, dil2 = dilation
    k1 = dil1 * (r1 - 1) + 1
    k2 = dil2 * (r2 - 1) + 1
    l1_out = (x.shape[-2] - k1) // stride1 + 1
    l2_out = (x.shape[-1] - k2) // stride2 + 1
    res = np.zeros(x.shape[:-3] + (c_out, l1_out, l2_out), x.dtype)
    for i, wi in enumerate(w.T):
        i_dilated = i * dil2
        i_slice = slice(i_dilated, i_dilated + l2_out * stride2, stride2)
        for j, wj in enumerate(wi):
            j_dilated = j * dil1
            j_slice = slice(j_dilated, j_dilated + l1_out * stride1, stride1)
            x_part = x[..., j_slice, i_slice]
            res += np.einsum('...ijk,il->...ljk', x_part, wj, optimize=True)
    return res


def sig2col(x, w_shape, stride=1, dilation=1):
    """
    Represent signal so that each 'column' represents
    the elements in a sliding window.

    Parameters
    ----------
    x : ndarray
        The signal to represent.
    w_shape : tuple of ints
        The shape of the window.
        The length defines the dimensionality.
    stride : int or tuple of ints, optional
        The stride(s) for each dimension of the window.
    dilation : int or tuple of ints, optional
        The dilation(s) for each dimension of the window.

    Returns
    -------
    cols : ndarray
        New representation of the array.
        This array will have `len(w_shape)` more dimensions than `x`.

    Notes
    -----
    This function implements the 'im2col' trick,
    used for implementing convolutions efficiently.
    """
    w_shape = np.asarray(w_shape)
    x_shape1, x_shape2 = np.split(x.shape, [-len(w_shape)])
    kernel_shape = dilation * (w_shape - 1) + 1
    out_shape2 = (x_shape2 - kernel_shape) // stride + 1

    # sliding window view (inspired by http://github.com/numpy/numpy/pull/10771)
    x_si1, x_si2 = np.split(x.strides, [len(x_shape1)])
    v_strides = tuple(x_si1) + tuple(stride * x_si2) + tuple(dilation * x_si2)
    v_shape = tuple(x_shape1) + tuple(out_shape2) + tuple(w_shape)
    _x = np.lib.stride_tricks.as_strided(x, v_shape, v_strides, writeable=False)
    return _x


def convolution_dot(w, x, stride=1, dilation=1):
    """
    Implements ND convolution as a dot product using the 'im2col' trick.

    Parameters
    ----------
    w : (Co, Ci, R1, R2, ..., Rn) ndarray
        Weights for the ND convolution.
    x : (N, Ci, L1, L2, ..., Ln) ndarray
        Input data for the ND convolution.
    stride : int or tuple of ints, optional
        Strides for the ND convolution.
        When providing a tuple of ints,
        the length should match the number of dimensions N.
    dilation : int or tuple of ints, optional
        Dilations for the ND convolution.
        When providing a tuple of ints,
        the length should match the number of dimensions N.

    Returns
    -------
    corr : (N, Co, L1', L2', ..., Ln') ndarray
        ND cross-correlation of `x` and `w`.
    """
    w = np.array(w, copy=False, ndmin=3)
    x = np.array(x, copy=False, ndmin=w.ndim - 1)
    stride = np.asarray(stride)
    dilation = np.asarray(dilation)

    nd = w.ndim - 2
    (c_out, c_in), w_shape2 = np.split(w.shape, [-nd])
    x_shape1, _ = np.split(x.shape, [-nd])

    if c_in != x_shape1[-1]:
        msg = "input channel dimensions do not match: {:d} != {:d}"
        raise ValueError(msg.format(c_in, x_shape1[-1]))

    x = sig2col(x, w_shape2, stride=stride, dilation=dilation)

    axes = [1] + [-i - 1 for i in reversed(range(nd))]
    res = np.tensordot(x, w, axes=(axes, axes))
    # assert -nd - 1 == 1, "code cannot be simplified"
    return np.moveaxis(res, -1, -nd - 1)
