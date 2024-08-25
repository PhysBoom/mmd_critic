import numpy as np
from .kernels import RBFKernel

class BaseMMD:
    def __init__(self, kernel="rbf", **kernel_params):
        """
        Initialize the BaseMMD class with the kernel and kernel parameters.

        Args:
            kernel (str): The type of kernel to use. Default is "rbf".
            kernel_params (dict): Additional parameters to pass to the kernel.
        """
        if kernel == "rbf":
            self.kernel = RBFKernel(**kernel_params)
        else:
            raise ValueError(f"Unsupported kernel type: {kernel}")

    def compute_mmd(self, X, Y, mean_K_XX=None):
        """
        Compute the squared MMD statistic between two datasets X and Y.

        Args:
            X: A matrix of shape (n_samples_X, n_features).
            Y: A matrix of shape (n_samples_Y, n_features).
            mean_K_XX (float or None): The precomputed mean of K_XX. If None, it will be computed.

        Returns:
            float: The computed squared MMD statistic.
        """
        X, Y = np.asarray(X), np.asarray(Y)

        if X.size == 0 and Y.size == 0:
            return 0.0
        elif X.size == 0:
            return np.mean(self.kernel(Y, Y))
        elif Y.size == 0:
            return mean_K_XX if mean_K_XX is not None else np.mean(self.kernel(X, X))

        if X.shape[1] != Y.shape[1]:
            raise ValueError("The number of features in X and Y must match.")

        K_XX = mean_K_XX if mean_K_XX is not None else np.mean(self.kernel(X, X))
        K_YY = np.mean(self.kernel(Y, Y))
        K_XY = np.mean(self.kernel(X, Y))

        sq_mmd = K_XX + K_YY - 2 * K_XY
        return sq_mmd


class MMD(BaseMMD):
    def __init__(self, kernel="rbf", **kernel_params):
        """
        Initialize the MMD class.

        Args:
            kernel (str): The type of kernel to use. Default is "rbf".
            kernel_params (dict): Additional parameters to pass to the kernel.
        """
        super().__init__(kernel, **kernel_params)

    def __call__(self, X, Y):
        """
        Compute the squared MMD statistic between two datasets X and Y.

        Args:
            X: A matrix of shape (n_samples_X, n_features).
            Y: A matrix of shape (n_samples_Y, n_features).

        Returns:
            float: The computed squared MMD statistic.
        """
        return self.compute_mmd(X, Y)


class CachedMMD(BaseMMD):
    def __init__(self, X, kernel="rbf", **kernel_params):
        """
        Initialize the CachedMMD class with dataset X and precompute mean(K_XX).

        Args:
            X (array-like): The dataset to be used for caching K_XX.
            kernel (str): The type of kernel to use. Default is "rbf".
            kernel_params (dict): Additional parameters to pass to the kernel.
        """
        super().__init__(kernel, **kernel_params)
        self.X = np.asarray(X)
        self.mean_K_XX = np.mean(self.kernel(self.X)) if self.X.size > 0 else 0.0

    def __call__(self, Y):
        """
        Compute the squared MMD statistic between the cached dataset X and a new dataset Y.

        Args:
            Y: A matrix of shape (n_samples_Y, n_features).

        Returns:
            float: The computed squared MMD statistic.
        """
        return self.compute_mmd(self.X, Y, mean_K_XX=self.mean_K_XX)