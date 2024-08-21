import numpy as np
from .kernels import RBFKernel

class MMD:
    def __init__(self, kernel="rbf", **kernel_params):
        """
        Initialize the MMD class.

        Args:
            kernel (str): The type of kernel to use. Default is "rbf".
            kernel_params (dict): Additional parameters to pass to the kernel.
        """
        if kernel == "rbf":
            self.kernel = RBFKernel(**kernel_params)
        else:
            raise ValueError(f"Unsupported kernel type: {kernel}")

    def __call__(self, X, Y):
        """
        Compute the squared MMD statistic between two datasets.

        Args:
            X: A matrix of shape (n_samples_X, n_features).
            Y: A matrix of shape (n_samples_Y, n_features).

        Returns:
            float: The computed squared MMD statistic.

        Note: The value computed here is the squared MMD rather than the MMD.
        """
        X = np.asarray(X)
        Y = np.asarray(Y)

        if X.shape[1] != Y.shape[1]:
            raise ValueError("The number of features in X and Y must match.")

        K_XX = self.kernel(X, X)
        K_YY = self.kernel(Y, Y)
        K_XY = self.kernel(X, Y)

        sq_mmd = np.mean(K_XX) + np.mean(K_YY) - 2 * np.mean(K_XY)        
        return sq_mmd
