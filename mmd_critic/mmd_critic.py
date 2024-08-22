from .mmd import MMD
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
        self.mmd = MMD(kernel, **kernel_params)
        self.kernel = self.mmd.kernel
        self.K_XX = self.kernel(self.X)

    def _A(self, S):
        """
        Computes A(S) as defined in Lemma 1 of the MMD Critic paper
        """
        n = self.X.shape[0]
        m = len(S)
        
        A = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                A[i, j] = (2 / (n * m)) * (j in S) - (1 / (m * m)) * ((i in S) and (j in S))
                
        return A
    
    def _prototype_cost(self, prototype_indices):
        return np.sum(self._A(prototype_indices) * self.K_XX)

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

        prototype_indices = []
        while len(prototype_indices) < n:
            cur_max = -math.inf
            cur_max_elem = -1
            for i in range(len(self.X)):
                print(i)
                if i in prototype_indices:
                    continue
                new_cost = self._prototype_cost(prototype_indices + [i])
                if new_cost > cur_max:
                    cur_max = new_cost
                    cur_max_elem = i
            prototype_indices.append(cur_max_elem)
        return np.asarray([self.X[i] for i in prototype_indices])
    
    def select_criticisms(self, n):
        """
        Greedily selects criticisms from the dataset in the class

        Args:
            n: number of criticisms to select

        Raises:
            ValueError on improper n
        """
        criticisms = []




