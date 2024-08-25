import numpy as np
import numexpr as ne
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
        
        X_norm = np.sum(X ** 2, axis=-1)
        Y_norm = np.sum(Y ** 2, axis=-1)
        XY_dot = np.dot(X, Y.T)
        
        gamma = 1 / (2 * self.sigma ** 2)
        K = ne.evaluate('exp(-g * (A + B - 2 * C))', {
            'A': X_norm[:, None],
            'B': Y_norm[None, :],
            'C': XY_dot,
            'g': gamma
        })
        return K
    
    def _compute_symmetric(self, X):
        X = np.asarray(X)
        
        X_norm = np.sum(X ** 2, axis=-1)
        XX_dot = np.dot(X, X.T)
        
        gamma = 1 / (2 * self.sigma ** 2)
        K = ne.evaluate('exp(-g * (A + B - 2 * C))', {
            'A': X_norm[:, None],
            'B': X_norm[None, :],
            'C': XX_dot,
            'g': gamma
        })
        return K
