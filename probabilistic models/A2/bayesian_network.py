from typing import List, Optional, Callable, Tuple, Iterator, Iterable
import numpy as np
import collections


def readonly_view(x: np.ndarray) -> np.ndarray:
    """
    Returns a view into the given numpy array
    that has the writeable flag set to False.
    """
    x = x.view()
    x.flags.writeable = False
    return x


class Variable:

    def __init__(self, pdt: np.ndarray, idx_mapping: Tuple[int]):
        """
        Creates a Variable object which is used to build Bayesian Networks.

        :param pdt: expanded and sorted (conditional) probability distribution table.
        :param idx_mapping: mapping of dimension to variable index.
         idx_mapping[0] == variable.id, idx_mapping[1:] == parents

        :returns: A Variable object.
        """
        # basic info
        assert len(idx_mapping) >= 1, 'Variable must have an id!'
        if idx_mapping[0] in idx_mapping[1:]:
            raise UserWarning(f'It makes no sense to condition on self e.g. P(A | A)! ID: {idx_mapping[0]}')
        self.id = idx_mapping[0]
        self.parents = frozenset(idx_mapping[1:])
        self.children = set()
        self._pdt = readonly_view(pdt)
        self.num_nodes = len(pdt.shape)
        self.num_values = pdt.shape[idx_mapping[0]]

        # resampling distribution, parents
        self.resampling_pdt = None
        self.resampling_parents = None

    @property
    def pdt(self):
        return self._pdt

    def __call__(self, sample: np.ndarray, resampling: bool = False):
        """
        Returns the probability distribution over the variable, given its parents or given its Markov blanket.

        :param sample: A NumPy array holding the values of the parent variables sorted by variable id.
                       Values of non-parent variables will be ignored.
        :param resampling: If False, P(X|pa(X)) will be returned. Otherwise P(X|mb(X)).
        :returns: A NumPy array representing the probability distribution over the variable,
                  given its parents or given its markov blanket.
        """

        assert len(sample) == self.num_nodes, f'Size of sample must be equal to number of variables in the Network. ' \
                                              f'Given: {len(sample)}, Expected: {self.num_nodes}'

        if resampling:
            assert self.resampling_parents is not None, 'Resampling distribution not computed!'
            parents = self.resampling_parents
            pdt = self.resampling_pdt
        else:
            parents = self.parents
            pdt = self.pdt

        index = ()
        for i in range(self.num_nodes):
            if i == self.id:
                index = index + (slice(None),)
            elif i in parents:
                index = index + (sample[i],)
            else:
                index = index + (0,)
        return pdt[index]
    
    def __hash__(self):
        return hash(self.id)


class BayesNet:

    def __init__(self, *pdt_ids_tuples: Tuple[np.ndarray, Iterable[int]],
                 resampling_distribution: Optional[
                     Callable[[Variable, 'BayesNet'],
                              Tuple[np.ndarray, List[int]]]] = None) -> None:
        """
        Creates a BayesNet object.

        :param pdt_ids_tuples: Arbitrarily many tuples in format (np.ndarray, [id1, id2, ...]).
            Each tuple defines one variable of the Bayesian Network. The numpy array stacks
            the Probability Distribution Tables (PDTs) of the variable conditioned on all value
            combinations of its parents. The integer list denotes the variable's id followed by
            its parent variable ids (if any), matching the order of dimensions in the PDTs.
            Each variable id is the index of the column in the data the variable corresponds to.
        :param resampling_distribution: Callable computing the resampling distribution given
            a variable and a BayesNet (Only needed in PS 3, Assignment 'Gibbs Sampling', and
            is described there thoroughly, completely ignore it otherwise).
        :return: The BayesNet object.
        """
        assert len(pdt_ids_tuples)>0, "Zero variables passed to BayesNet."
        
        for t in pdt_ids_tuples:
            assert isinstance(t, tuple), f"Passed variable descriptors must be tuples but got {type(t)}. Did you use the asterisk (*) to unpack the list of variable descriptors into the parametes of the function call, e.g., BayesNet(*zip(pdts, indices))?"
            assert isinstance(t[0], np.ndarray), f"First element of variable descriptor must be a np.ndarray but got {type(t[0])}. Received tuple: {t}. Did you call BayesNet(*zip(pdts, indices))?"
            assert isinstance(t[1], collections.abc.Iterable), f"Second element of variable descriptor must be a Iterable (e.g., list or tuple) but got {type(t[1])}. Received tuple: {t}. Did you call BayesNet(*zip(pdts, indices))?"
            assert len(t) == 2, f"Tuple must contain exactly two elements but got {len(t)}. Tuple: {t}"
        
        self.nodes = dict()
        self._pdts, self._indices = zip(*pdt_ids_tuples)
        self._pdts = tuple(readonly_view(pdt) for pdt in self._pdts)
        num_nodes = len(self.pdts)

        for pdt, structure in zip(self.pdts, self.indices):
            assert set(structure).issubset(set(range(num_nodes))), f"Invalid node ID in table descriptor {structure}. Node IDs must be in range(num_nodes) ( < {len(pdt_ids_tuples)}). Tuple: {pdt, structure}."
            assert type(pdt) == np.ndarray, f'Probability Density Table has to be a NumPy ndarray' \
                                            f' but was of type {type(pdt)}!'
            assert np.all(np.isclose(pdt.sum(axis=0), 1)), f'Probabilities on axis 0 have to sum to 1!'
            assert pdt.ndim == len(
                structure), f'Number of table dimensions has to match ' \
                            f'the number of Variable indices (1 (self) + n_parents)!' \
                            f'N-Dimensions: {pdt.ndim} != Len(Idcs): {len(structure)}!'
            # Order PDT dimensions by variable id
            pdt = pdt.transpose(np.argsort(structure))
            # Add singleton dimensions for all other variables
            to_expand = tuple(set(range(num_nodes)).difference(set(structure)))
            pdt = np.expand_dims(pdt, axis=to_expand)
            # Create Variable object
            assert self.nodes.get(structure[0]) is None, f"Duplicate variable descriptor for variable {structure[0]}."
            self.nodes[structure[0]] = Variable(pdt, structure)

        # Set children of nodes
        for node in self.nodes.values():
            for parent_id in node.parents:
                self.nodes[parent_id].children.add(node.id)

        # Compute resampling distributions
        if resampling_distribution is not None:
            for node in self.nodes.values():
                node.resampling_pdt, node.resampling_parents = resampling_distribution(node, self)
                node.resampling_pdt = readonly_view(node.resampling_pdt)

    @property
    def pdts(self):
        return self._pdts

    @property
    def indices(self):
        return self._indices

    def __len__(self) -> int:
        """
        Retrieves the number of Variables in the network.

        :return: Variable count as integer
        """
        return len(self.nodes)

    def __getitem__(self, id: int) -> Variable:
        """
        Retrieves a Variable based on its id.

        :param id: Id of the Variable.
        :return: A BayesNet Variable Object with the corresponding id.
        :raises KeyError: if id is not found.
        """
        return self.nodes[id]

    def __iter__(self) -> Iterator[Variable]:
        """
        Iterates over all variables in the bayesian network in topological ordering, i.e.,
        for an edge from a variable X to a variable Y, X is returned before Y.  
        Since a bayesian network is a directed acyclic graph, a topological ordering can always be found. 

        :yields: variable after variable according to the network's topology.
        """
        
        raise NotImplementedError("Topological Sort for BayesNet class is to be implemented in Problem Set 2!")

