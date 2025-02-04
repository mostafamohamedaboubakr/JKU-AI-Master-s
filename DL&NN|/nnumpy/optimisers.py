import numpy as np


__all__ = ["Optimiser", "GradientDescent", "GD",
           "ResilientPropagation", "RProp",
           "ResilientMeanSquaresPropagation", "RMSProp",
           "Adam", "Adamax"]


class Optimiser:
    """ Base class for NNumpy optimisers. """

    def __init__(self, parameters, lr: float):
        """
        Create an optimiser instance.

        Parameters
        ----------
        parameters : iterable
            Iterable over the parameters that need to be updated.
        lr : float
            Learning rate or step size for updating the parameters.
        """
        self.parameters = list(parameters)
        if len(self.parameters) == 0:
            raise ValueError("no parameters to optimise")

        self.lr = float(lr)
        if self.lr < 0.:
            raise ValueError("learning rate must be positive")

        self.state = [self.init_state(par) for par in self.parameters]

    def init_state(self, par):
        """
        Create the initial optimiser state for a parameter.

        Parameters
        ----------
        par : Parameter
            The parameter to create the initial state for.

        Returns
        -------
        state : object
            The initial optimiser state for the given parameter.
        """
        return None

    def step(self):
        """
        Update all parameters under control of this optimiser
        by making one step in the update direction
        as computed by this algorithm for each of the parameters.
        """
        new_states = []
        for w, state in zip(self.parameters, self.state):
            delta_w, new_state = self.get_direction(w.grad, state)
            w -= self.lr * delta_w
            del w.grad  # safeguard
            new_states.append(new_state)

        self.state = new_states

    def get_direction(self, grad, state):
        """
        Compute the update direction from gradient and state for single parameter.

        Parameters
        ----------
        grad : ndarray
            Gradient direction.
        state : object or tuple of objects
            State information that is necessary to compute the update direction.

        Returns
        -------
        delta_w : ndarray
            The update direction according to the algorithm.
        new_state: object or tuple of objects
            Updated state information after computing the update direction.
        """
        raise NotImplementedError("method must be implemented in subclass")


class GradientDescent(Optimiser):
    """ NNumpy implementation of gradient descent. """

    def __init__(self, parameters, lr: float, momentum: float = 0.):
        """
        Parameters
        ----------
        momentum : float
            Momentum term for the gradient descent.
        """
        super().__init__(parameters, lr)
        self.mu = float(momentum)
        if self.mu < 0.:
            raise ValueError("momentum term must be positive")

    def init_state(self, par):
        return np.zeros_like(par)

    def get_direction(self, grad, state):
        state = state * self.mu + (1 - self.mu) * grad
        return state, state


class ResilientPropagation(Optimiser):
    """ NNumpy implementation of the RProp algorithm. """

    def __init__(self, parameters, lr: float, factors: tuple = (.5, 1.2),
                 lr_limits: tuple = (1e-6, 50)):
        """
        Parameters
        ----------
        factors : tuple of 2 floats, optional
            Factors to scale learning rates down, resp. up.
        lr_limits : tuple of 2 floats, optional
            Lower resp. upper bound for the learning rates.
        """
        super().__init__(parameters, lr)

        eta_min, eta_plus = sorted(factors)
        self.eta_min = float(eta_min)
        self.eta_plus = float(eta_plus)
        eta_low, eta_high = sorted(lr_limits)
        self.eta_low = float(eta_low)
        self.eta_high = float(eta_high)

    def init_state(self, par):
        return np.ones_like(par), np.zeros_like(par)

    def get_direction(self, grad, state):
        lr_scale, dw_old = state
        sign_change = np.sign(grad * dw_old)

        etas = np.where(sign_change < 0, self.eta_min, self.eta_plus)
        etas = np.where(sign_change == 0, 0, etas)
        lr_scale *= np.clip(etas, self.eta_low, self.eta_high)
        dw_old = np.multiply(dw_old, sign_change >= 0, out=dw_old)
        return dw_old, (lr_scale, dw_old)


class AdaGrad(Optimiser):
    """ NNumpy implementation of the AdaGrad algorithm. """

    def __init__(self, parameters, lr: float, epsilon: float = 1e-7):
        """
        Parameters
        ----------
        epsilon : float, optional
            Small number that is added to denominator for numerical stability.
        """
        super().__init__(parameters, lr)
        self.eps = float(epsilon)

    def init_state(self, par):
        return np.zeros_like(par)

    def get_direction(self, grad, state):
        state += grad ** 2
        return grad / (np.sqrt(state) + self.eps), state


class ResilientMeanSquaresPropagation(Optimiser):
    """
    NNumpy implementation of the generalisation to the RProp algorithm
    that uses Mean Squares to approximate the sign function over mini-batches.
    """

    def __init__(self, parameters, lr: float = 1e-3, rho: float = .5,
                 epsilon: float = 1e-7):
        """
        Parameters
        ----------
        rho : float, optional
            Decay factor for the exponential averaging of mean squares.
        epsilon : float, optional
            Small number that is added to denominator for numerical stability.
        """
        super().__init__(parameters, lr)
        self.rho = float(rho)
        self.eps = float(epsilon)

    def init_state(self, par):
        return np.ones_like(par)

    def get_direction(self, grad, state):
        state *= self.rho
        state += (1 - self.rho) * grad ** 2
        return grad / (np.sqrt(state) + self.eps), state


class Adam(Optimiser):
    """ NNumpy implementation of the Adam algorithm. """

    def __init__(self, parameters, lr: float = 1e-3, betas: tuple = (.9, .999),
                 epsilon: float = 1e-7, bias_correction=True):
        """
        Parameters
        ----------
        betas : tuple of 2 floats, optional
            Decay factors for the exponential averaging of mean, resp. variance.
        epsilon : float, optional
            Small number that is added to denominator for numerical stability.
        bias_correction : bool, optional
            Whether or not mean and bias estimates should be bias-corrected.
        """
        super().__init__(parameters, lr)

        beta1, beta2 = betas
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(epsilon)
        self.bias_correction = bias_correction

    def init_state(self, par):
        return np.zeros_like(par), np.zeros_like(par), 1

    def get_direction(self, grad, state):
        m, v, step_count = state
        m *= self.beta1
        m += (1 - self.beta1) * grad
        v *= self.beta2
        v += (1 - self.beta2) * grad ** 2

        mean, var = np.copy(m), np.copy(v)
        if self.bias_correction:
            mean /= (1 - self.beta1 ** step_count)
            var /= (1 - self.beta2 ** step_count)

        return mean / (np.sqrt(var) + self.eps), (m, v, step_count + 1)


class Adamax(Optimiser):
    """ NNumpy implementation of the Adamax algorithm. """

    def __init__(self, parameters, lr: float = 1e-3, betas: tuple = (.9, .999),
                 epsilon: float = 1e-7, bias_correction=True):
        """
        Parameters
        ----------
        betas : tuple of 2 floats, optional
            Decay factors for the exponential averaging of mean, resp. variance.
        epsilon : float, optional
            Small number that is added to denominator for numerical stability.
        bias_correction : bool, optional
            Whether or not mean and bias estimates should be bias-corrected.
        """
        super().__init__(parameters, lr)

        beta1, beta2 = betas
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(epsilon)
        self.bias_correction = bias_correction

    def init_state(self, par):
        return np.zeros_like(par), np.zeros_like(par), 1

    def get_direction(self, grad, state):
        m, u, step_count = state
        m *= self.beta1
        m += (1 - self.beta1) * grad
        u = np.maximum(self.beta2 * u, np.abs(grad) + self.eps, out=u)

        mean = np.copy(m)
        if self.bias_correction:
            mean /= (1 - self.beta1 ** step_count)

        return mean / u, (m, u, step_count + 1)


# shorter names for optimisers
GD = GradientDescent
RProp = ResilientPropagation
RMSProp = ResilientMeanSquaresPropagation
