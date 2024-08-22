from .mmd import CachedMMD
import numpy as np
import math

class MMDCritic:
    """
    MMDCritic is a class designed to facilitate prototype and criticism selection using the MMD Critic method

    Attributes:
        X (array-like): The dataset for which to draw prototypes.
        mmd (MMD): An instance of the MMD class, initialized with a specified kernel and kernel parameters.
    """
    def __init__(self, X, kernel="rbf", **kernel_params):
        """
        Initializes the MMD Critic class

        Args:
            X (array-like): The dataset
            kernel (str): The kernel to use. Currently, only rbf is supported
            kernel_params: Arguments to pass to the kernel function
        """
        self.X = np.asarray(X)
        self.mmd = CachedMMD(X, kernel, **kernel_params)
        self._prototype_additive_bias = self.mmd([])

    def _prototype_cost(self, S):
        """Computes the cost function for a given subset of prototypes S"""
        return self._prototype_additive_bias - self.mmd(S)

    def select_prototypes(self, n):
        """
        Greedily selects prototypes from the dataset in the class

        Args:
            n: number of prototypes to select

        Raises:
            ValueError on improper n
        """
        if n > len(self.X) or n <= 0:
            raise ValueError("n must satisfy 0 < n <= len(X)")

        prototypes = []
        selected_indices = set()
        while len(prototypes) < n:
            cur_max = -math.inf
            cur_max_elem = self.X[0]
            cur_max_index = 0
            for i, dp in enumerate(self.X):
                if i in selected_indices:
                    continue
                new_cost = self._prototype_cost(prototypes + [dp] if prototypes else [dp])
                if new_cost > cur_max:
                    cur_max = new_cost
                    cur_max_elem = dp
                    cur_max_index = i
            prototypes.append(cur_max_elem)
            selected_indices.add(cur_max_index)
            print("Selected prototype", len(prototypes))
        return np.asarray(prototypes)
    
    def select_criticisms(self, n):
        """
        Greedily selects criticisms from the dataset in the class

        Args:
            n: number of criticisms to select

        Raises:
            ValueError on improper n
        """
        criticisms = []




