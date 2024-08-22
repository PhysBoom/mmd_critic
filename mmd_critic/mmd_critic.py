from .mmd import MMD
import numpy as np

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
        self.K = self.kernel(self.X)

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

        selected_indices = self._greedy_select_protos(self.K, np.arange(len(self.X)), n)
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
        """
        if len(candidate_indices) != np.shape(K)[0]:
            K = K[:, candidate_indices][candidate_indices, :]

        n = len(candidate_indices)
        colsum = 2 * np.sum(K, axis=0) / n

        selected = np.array([], dtype=int)
        for i in range(m):
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
    
    def select_criticisms(self, n):
        """
        Greedily selects criticisms from the dataset in the class

        Args:
            n: number of criticisms to select

        Raises:
            ValueError on improper n
        """
        criticisms = []




