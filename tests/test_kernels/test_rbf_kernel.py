import unittest
import numpy as np
from mmd_critic.kernels.rbf_kernel import RBFKernel

class TestRBFKernel(unittest.TestCase):
    def test_rbf_1_dim(self):
        rbf = RBFKernel(sigma=1)
        res = rbf(np.asarray([[1], [2]]), np.asarray([[3], [4]]))
        self.assertAlmostEqual(res[0][0], np.exp(-2))
        self.assertAlmostEqual(res[0][1], np.exp(-9/2))

    def test_rbf_1_dim_diff_samples(self):
        rbf = RBFKernel(sigma=0.5)
        res = rbf([[1]], [[1], [2]])
        self.assertEqual(res.shape, (1, 2))
        self.assertAlmostEqual(res[0][0], 1)
        self.assertAlmostEqual(res[0][1], np.exp(-2))

    def test_rbf_multidim(self):
        rbf = RBFKernel()
        res = rbf(np.asarray([[1, 2], [3, 4]]), np.asarray([[5, 6]]))
        self.assertAlmostEqual(res[0][0], np.exp(-16))
        self.assertAlmostEqual(res[1][0], np.exp(-4))

    def test_error_on_negative_sigma(self):
        with self.assertRaises(ValueError):
            RBFKernel(sigma=-1)

    def test_error_on_zero_sigma(self):
        with self.assertRaises(ValueError):
            RBFKernel(sigma=0)

    def test_error_on_improper_shape(self):
        with self.assertRaises(ValueError):
            RBFKernel()(np.asarray([[[1]]]), np.asarray([[[1]]]))

    def test_error_on_dimension_mismatch(self):
        with self.assertRaises(ValueError):
            RBFKernel()(
                np.asarray([[1, 2], [3, 4]]),
                np.asarray([[1, 2, 3], [4, 5, 6]])
            )
