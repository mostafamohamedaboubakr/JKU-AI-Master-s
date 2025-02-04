import operator

import numpy as np

from nnumpy.base import Module

__all__ = ['Container', 'Sequential']


class Container(Module):
    """Base class for NNumpy modules with submodules."""

    def __init__(self):
        super().__init__()
        self._modules = []
        self._name_index = {}

    # # # attribute stuff # # #

    def __dir__(self):
        yield from super().__dir__()
        yield from self._name_index.keys()

    def __getattr__(self, name):
        try:
            idx = self._name_index[name]
            return self._modules[idx]
        except KeyError:
            return super().__getattr__(name)

    def __setattr__(self, name, value):
        if value is self:
            raise ValueError("adding a module to itself is not allowed")

        try:
            _name_index = self.__dict__.get('_name_index', {})
            idx = _name_index[name]
            self._modules[idx] = value
        except KeyError:
            if isinstance(value, Module):
                return self.add_module(value, name=name)

            super().__setattr__(name, value)

    def __delattr__(self, name):
        try:
            return self.pop_module(name)
        except KeyError:
            return super().__delattr__(name)

    # # # list-like stuff # # #

    def __getitem__(self, index):
        return self._modules[index]

    def __setitem__(self, index, module):
        if isinstance(index, slice):
            raise NotImplementedError("sliced assignment not implemented")
        self._modules[index] = module

    def __delitem__(self, index):
        if isinstance(index, slice):
            iterable = range(index.start or 0, index.stop or -1, index.step or 1)
            for idx in iterable:
                self.pop_module(idx)
        else:
            self.pop_module(index)

    def __len__(self):
        return self._modules.__len__()

    def __iter__(self):
        return self._modules.__iter__()

    def __reversed__(self):
        return self._modules.__reversed__()

    # # # non-magic stuff # # #

    def get_state(self, parameters=True):
        """
        Get an immutable representation of the state of this module.

        Returns
        -------
        state : dict
            A dictionary with all data necessary to restore the module
            to the current state. Subsequent changes to the module
            will not be reflected in the returned state.
        parameters : bool, optional
            Whether or not to include the registered parameters in the state.

        """
        return {name: mod.get_state(parameters)
                for name, mod in self.named_modules()}

    def set_state(self, state):
        """
        Put this module in a particular state.

        Parameters
        ----------
        state : dict
            State object as returned from `get_state`.

        Notes
        -----
        Subsequent changes to the module will not be reflected
        in the state object that is passed on.

        """
        for name, mod in self.named_modules():
            mod.set_state(state.pop(name, {}))

    def train(self):
        """ Put the module in training mode. """
        for mod in self._modules:
            mod.train()

        self.predicting = False
        return self

    def eval(self):
        """ Put the module in evaluation mode. """
        for mod in self._modules:
            mod.eval()

        self.predicting = True
        return self

    def add_module(self, module, name=None):
        """
        Add a submodule with its parameters to this container.

        Parameters
        ----------
        module : Module
            Module object.
        name : str, optional
            Name of the submodule.
        """
        if module is self:
            raise ValueError("adding a module to itself is not allowed")
        if name is not None:
            self._name_index[name] = len(self._modules)
        self._modules.append(module)
        return module

    def pop_module(self, identifier=-1):
        """
        Remove submodule with its parameters from this container.

        Parameters
        ----------
        identifier : str or int, optional
            Name or index of the submodule.
            If identifier is None, the last submodule is removed.

        Returns
        -------
        module : Module
            The removed submodule.
        """
        try:
            idx = operator.index(identifier)
        except TypeError:
            name = str(identifier)
            idx = self._name_index.pop(name)

        module = self._modules.pop(idx)
        self._name_index = {k: v if v < idx else v - 1
                            for k, v in self._name_index}

        return module

    def named_modules(self):
        """
        Iterator over submodule (name, module) pairs.

        Yields
        -------
        name : str
            Name of the module in this container.
        mod : Module
            Submodule of this container.
        """
        index_name = {v: k for k, v in self._name_index.items()}
        for idx, mod in enumerate(self._modules):
            m_name = index_name.get(idx, "({:d})".format(idx))
            yield m_name, mod

    def named_parameters(self):
        yield from super().named_parameters()
        for m_name, mod in self.named_modules():
            for p_name, par in mod.named_parameters():
                yield '.'.join([m_name, p_name]), par

    def reset_parameters(self, seed: int = None, **kwargs):
        rng = np.random.default_rng(seed)
        for name, mod in self.named_modules():
            mod.reset_parameters(seed=rng.integers(1 << 32), **{
                k.split(f"{name}.", maxsplit=1)[-1]: v for k, v in kwargs.items()
            })

    def compute_outputs(self, *inputs):
        raise NotImplementedError()

    def compute_grads(self, grads, cache):
        raise NotImplementedError()


class Sequential(Container):
    """
    NNumpy module that chains together multiple one-to-one sub-modules.

    Examples
    --------
    Doubling a module could be done as follows:
    >>> module = Module()
    >>> seq = Sequential(module, module)

    Modules can be accessed by index or by iteration:
    >>> assert module is seq[0] and module is seq[1]
    >>> mod1, mod2 = (m for m in seq)
    >>> assert mod1 is module and mod2 is module
    """

    def __init__(self, *modules):
        super().__init__()
        for i, mod in enumerate(modules):
            self.add_module(mod, name="module" + str(i))

    def compute_outputs(self, x):
        cached = []
        for module in self:
            x, cache = module.compute_outputs(x)
            cached.insert(0, cache)

        return x, cached

    def compute_grads(self, grads, cache):
        for module, c in zip(reversed(self), cache):
            grads = module.compute_grads(grads, c)

        return grads
