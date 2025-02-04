import numpy as np

__all__ = ["Distribution", "Uniform", "Gaussian", "Binomial"]


class Distribution:
    """ NNumpy base class for distributions. """

    def __init__(self, seed: int = None):
        self.rng = np.random.default_rng(seed)

    @property
    def mean(self):
        raise NotImplementedError("subclass must implement this method")

    @mean.setter
    def mean(self, value):
        raise NotImplementedError("subclass must implement this method")

    @property
    def var(self):
        raise NotImplementedError("subclass must implement this method")

    @var.setter
    def var(self, value):
        raise NotImplementedError("subclass must implement this method")

    def sample(self, shape):
        raise NotImplementedError("subclass must implement this method")


class Uniform(Distribution):
    """ NNumpy implementation of a uniform distribution. """

    def __init__(self, low: float = 0., high: float = 1., seed: int = None):
        super().__init__(seed)
        if low >= high:
            raise ValueError("lower bound must be less than upper bound")

        self.low = float(low)
        self.high = float(high)

    @property
    def mean(self):
        return (self.low + self.high) / 2

    @mean.setter
    def mean(self, value: float):
        radius = (self.high - self.low) / 2
        self.low = value - radius
        self.high = value + radius

    @property
    def var(self):
        return (self.high - self.low) ** 2 / 12

    @var.setter
    def var(self, value: float):
        if value < 0:
            raise ValueError("can not set variance to negative value")

        mean = self.mean
        self.low = mean - np.sqrt(3 * value)
        self.high = mean + np.sqrt(3 * value)

    def sample(self, shape: tuple):
        return self.rng.uniform(self.low, self.high, shape)


class Gaussian(Distribution):
    """ NNumpy implementation of a Gaussian distribution. """

    def __init__(self, avg=0., var=1., seed: int = None):
        super().__init__(seed)
        self.mean = avg
        self.var = var

    def pdf(self, x):
        raw = np.exp(- (x - self.mean) ** 2 / (2. * self.var))
        return raw / np.sqrt(2. * np.pi * self.var)

    def cdf(self, x):
        return (1. + np.erf((x - self.mean) / np.sqrt(self.var))) / 2.

    @property
    def mean(self):
        return self._avg

    @mean.setter
    def mean(self, value: float):
        self._avg = float(value)

    @property
    def var(self):
        return self._var

    @var.setter
    def var(self, value: float):
        if value < 0:
            raise ValueError("can not set variance to negative value")

        self._var = float(value)

    def sample(self, shape: tuple):
        return self.rng.normal(self._avg, np.sqrt(self._var), shape)


class Binomial(Distribution):
    """ NNumpy implementation of a Binomial distribution. """

    def __init__(self, n: int = 1, p: float = .5, seed: int = None):
        super().__init__(seed)
        if n < 0:
            raise ValueError("number of trials must be positive")
        if p < 0. or p > 1.:
            raise ValueError("probability must be value between zero and one")

        self.n = int(n)
        self.p = float(p)

    @property
    def mean(self):
        return self.n * self.p

    @mean.setter
    def mean(self, value: float):
        if value <= self.var:
            raise ValueError("can not set mean to value less than variance")

        p = 1 - self.var / value
        self.n = int(value / p)
        self.p = value / self.n

    @property
    def var(self):
        return self.n * self.p * (1 - self.p)

    @var.setter
    def var(self, value: float):
        mean = self.mean
        if value >= mean:
            raise ValueError("can not set variance to value greater than mean")

        p = 1 - value / mean
        self.n = int(mean / p)
        self.p = mean / self.n

    def sample(self, shape: tuple):
        return self.rng.binomial(self.n, self.p, shape)
