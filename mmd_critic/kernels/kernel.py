from abc import ABC, abstractmethod
import numpy as np

class Kernel(ABC):
    """
    An abstract base class for kernel functions.

    This class provides an interface for kernel functions used in various 
    machine learning algorithms, such as support vector machines, Gaussian 
    processes, or other kernel-based methods.

    Subclasses should implement the `_compute` method, which defines the 
    specific kernel function.
    """

    def __call__(self, X, Y):
        """
        Makes the Kernel instance callable. Verifies that the input types are 
        numpy arrays and calls the private `_compute` method.

        Args:
            X (np.ndarray): An array of shape (n_samples_X, n_features).
            Y (np.ndarray): An array of shape (n_samples_Y, n_features).

        Returns:
            np.ndarray: The computed kernel matrix of shape (n_samples_X, n_samples_Y).

        Raises:
            ValueError on improper arrays
        """
        X, Y = np.asarray(X), np.asarray(Y)
        if X.ndim != 2 or Y.ndim != 2:
            raise ValueError("Both X and Y must be 2D arrays.")
        
        if X.shape[1] != Y.shape[1]:
            raise ValueError("The number of features in X and Y must match.")
        
        return self._compute(X, Y)

    @abstractmethod
    def _compute(self, X, Y):
        """
        Computes the kernel matrix between two sets of vectors.

        Args:
            X (np.ndarray): An array of shape (n_samples_X, n_features).
            Y (np.ndarray): An array of shape (n_samples_Y, n_features).

        Returns:
            np.ndarray: The computed kernel matrix of shape (n_samples_X, n_samples_Y).
        
        Subclasses must implement this method to define the specific kernel function.
        """
        pass
