import math
from .mmd import MMD
import numpy as np
import random

class MMDCritic:
    """
    MMDCritic is a class designed to facilitate prototype and criticism selection using the MMD Critic method

    Attributes:
        X (array-like): The dataset for which to draw prototypes.
        mmd (MMD): An instance of the MMD class, initialized with a specified kernel and kernel parameters.
    """
    def __init__(self, X, prototype_kernel, criticism_kernel=None):
        """
        Initializes the MMD Critic class

        Args:
            X (array-like): The dataset
            prototype_kernel (Kernel): The kernel to use for prototypes. Should extend the mmd_critic.kernels.Kernel class
            criticism_kernel (Optional[Kernel]): The kernel to use for criticisms. If None, uses the prototype kernel
        """
        self.X = np.asarray(X)
        self.prototype_kernel = prototype_kernel
        self.criticism_kernel = criticism_kernel or prototype_kernel
        self.prot_K = self.prototype_kernel(X)
        self.crit_K = self.prot_K if not criticism_kernel else self.criticism_kernel(X)

    def select_prototypes(self, n):
        """
        Greedily selects prototypes from the dataset using the efficient method.

        Args:
            n: Number of prototypes to select.

        Raises:
            ValueError on improper n.
        """
        if n > len(self.X) or n <= 0:
            raise ValueError("n must satisfy 0 < n <= len(X)")

        selected_indices = self._greedy_select_protos(self.prot_K, np.arange(len(self.X)), n)
        return self.X[selected_indices]

    def _greedy_select_protos(self, K, candidate_indices, m):
        """
        Efficiently selects prototypes using the greedy algorithm.

        Args:
            K (ndarray): The kernel matrix.
            candidate_indices (ndarray): The indices of candidate prototypes.
            m (int): The number of prototypes to select.

        Returns:
            ndarray: The indices of the selected prototypes.

        Note: This uses Been Kim's original implementation, found at
        https://github.com/BeenKim/MMD-critic. It is more efficient than a more
        Naive approach, albeit counterintuitive.
        """
        if len(candidate_indices) != K.shape[0]:
            K = K[:, candidate_indices][candidate_indices, :]

        n = len(candidate_indices)
        colsum = 2 * np.sum(K, axis=0) / n

        selected = np.array([], dtype=int)
        for _ in range(m):
            candidates = np.setdiff1d(range(n), selected)
            s1array = colsum[candidates]

            if len(selected) > 0:
                temp = K[selected, :][:, candidates]
                s2array = np.sum(temp, axis=0) * 2 + np.diagonal(K)[candidates]
                s2array = s2array / (len(selected) + 1)
                s1array -= s2array
            else:
                s1array -= np.abs(np.diagonal(K)[candidates])

            argmax = candidates[np.argmax(s1array)]
            selected = np.append(selected, argmax)

        return candidate_indices[selected]
    
    def select_criticisms(self, n, protos, regularization:str | None="logdet"):
        """
        Greedily selects criticisms from the dataset in the class

        Args:
            n: number of criticisms to select
            protos: the prototypes. Must have protos as a subset of X
            regularization: the regularization to use. Only logdet and none are supported for now

        Raises:
            ValueError on improper n, reg
        """
        num_dp = self.X.shape[0]

        if regularization not in [None, "logdet"]:
            raise ValueError(f"Improper regularization {regularization}. Only 'logdet' and None are supported for now")
        
        if n <= 0 or n > num_dp - len(protos):
            raise ValueError(f"Must have 0 < n <= len(X) - len(protos)")
        
        proto_indices = [i for i in range(len(self.X)) if any(np.array_equal(self.X[i], proto) for proto in protos)]

        # We shouldnt but could have a data point twice, however we should never
        # have less indices than protos
        if len(proto_indices) < len(protos):
            raise ValueError("Prototypes are not a subset of X!")

        selected = np.array([], dtype=int)
        candidates = np.setdiff1d(range(num_dp), proto_indices)
        colsum = np.sum(self.crit_K, axis=0)/num_dp

        for _ in range(n):
            candidates = np.setdiff1d(candidates, selected)
            scores = np.abs(colsum[candidates] - np.sum(self.crit_K[proto_indices, :][:, candidates], axis=0)/len(protos))

            if regularization == "logdet":
                scores -= np.log(np.abs(np.diagonal(self.crit_K)[candidates]))
            selected = np.append(selected, candidates[np.argmax(scores)])
        return self.X[selected]












