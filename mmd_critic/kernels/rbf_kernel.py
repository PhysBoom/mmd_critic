import numpy as np
from .kernel import Kernel

class RBFKernel(Kernel):
    def __init__(self, sigma=1.0):
        """
        Radial Basis Function (RBF) kernel.

        Args:
            sigma (float): The bandwidth parameter (must be positive).
        """
        if sigma <= 0:
            raise ValueError("Sigma must be positive.")
        self.sigma = sigma

    def _compute(self, X, Y=None):
        X = np.asarray(X)
    
        if Y is None:
            return self._compute_symmetric(X)
        
        Y = np.asarray(Y)
        
        # Efficiently compute square pairwise Euclidean distances
        X_norm = np.sum(X ** 2, axis=1).reshape(-1, 1)
        Y_norm = np.sum(Y ** 2, axis=1).reshape(1, -1)
        dist = X_norm + Y_norm - 2 * np.dot(X, Y.T)
        
        K = np.exp(-dist / (2 * self.sigma ** 2))        
        return K
    
    def _compute_symmetric(self, X):
        X = np.asarray(X)
        
        X_norm = np.sum(X ** 2, axis=1).reshape(-1, 1)
        dist = X_norm + X_norm.reshape(1, -1) - 2 * np.dot(X, X.T)
        
        K = np.exp(-dist / (2 * self.sigma ** 2))        
        return K
