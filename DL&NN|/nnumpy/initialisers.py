import numpy as np

from .distributions import Distribution, Uniform, Gaussian, Binomial


class Initialiser:
    """ NNumpy base class for weight initialisation techniques. """

    def __init__(self, gain=1.):
        self.gain = float(gain)

    def __call__(self, *args):
        gain_sqrt = self.gain ** .5
        for arg in args:
            values = self.get_values(arg.shape)
            np.copyto(arg, gain_sqrt * values)

    def get_values(self, shape):
        raise NotImplementedError("subclass must implement this method")


class Constant(Initialiser):
    """ NNumpy implementation of constant initialisation. """

    def __init__(self, value: float, gain: float = 1.):
        super().__init__(gain)
        self.value = float(value)

    def get_values(self, shape):
        return np.broadcast_to(self.value, shape)


class Diagonal(Initialiser):
    """ NNumpy implementation of initialisation with diagonal tensor. """

    def __init__(self, value: float, gain: float = 1.):
        super().__init__(gain)
        self.value = float(value)

    def get_values(self, shape):
        data = np.zeros(shape)
        step = np.sum(np.cumprod(shape[:0:-1]))
        data[:min(shape)].flat[::step + 1] = self.value
        return data


class Random(Initialiser):
    """ NNumpy implementation of random initialisation. """

    def __init__(self, dist: Distribution = Uniform(-1, 1), gain: float = 1.):
        super().__init__(gain)
        self.dist = dist

    def get_values(self, shape):
        return self.dist.sample(shape)


class LeCun(Random):
    """ NNumpy implementation of LeCun random initialisation. """

    def get_values(self, shape):
        self.dist.var = 1 / shape[0]
        return super().get_values(shape)


class Glorot(Random):
    """ NNumpy implementation of Glorot random initialisation. """

    def get_values(self, shape):
        self.dist.var = 2 / (shape[0] + shape[1])
        return super().get_values(shape)


class Orthogonal(Random):
    """ NNumpy implementation of Orthogonal random initialisation. """

    def get_values(self, shape):
        samples = super().get_values(shape)
        samples = samples.reshape(shape[0], -1)
        num_rows, num_cols = samples.shape

        if num_rows < num_cols:
            samples = samples.T

        q, r = np.linalg.qr(samples)
        q *= np.sign(np.diag(r))  # uniform eigenspectrum

        if num_rows < num_cols:
            q = q.T

        return q.reshape(shape)


_normal_dist = Gaussian()
_uniform_dist = Uniform(-1., 1.)
_binomial_dist = Binomial()

zeros = Constant(value=0.)
ones = Constant(value=1.)
identity = Diagonal(value=1.)
uniform = Random(_uniform_dist)
normal = Random(_normal_dist)
binary = Random(_binomial_dist)
lecun_uniform = LeCun(_uniform_dist)
lecun_normal = LeCun(_normal_dist)
glorot_uniform = LeCun(_uniform_dist)
glorot_normal = Glorot(_normal_dist)
he_uniform = LeCun(_uniform_dist, gain=2.)
he_normal = LeCun(_normal_dist, gain=2.)
orthogonal = Orthogonal(_normal_dist)
