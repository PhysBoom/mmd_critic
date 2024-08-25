import unittest
from mmd_critic.mmd import MMD, CachedMMD

class TestMMD(unittest.TestCase):
    def test_proper_mmd_1(self):
        mmd = MMD()
        self.assertAlmostEqual(
            mmd([[1, 2], [3 ,4]], [[5, 6]]),
            1.4908,
            4
        )

    def test_proper_mmd_2(self):
        mmd = MMD(kernel="rbf", sigma=0.5)
        self.assertAlmostEqual(
            mmd([[1]], [[1], [2], [3]]),
            0.6364,
            4
        )

    def test_symmetry(self):
        X = [[1, 2, 3, 4], [5, 6, 7, 8], [1, 4, 6, 5]]
        Y = [[1, 5, 3, 4], [2, 5, 3, 4]]
        mmd = MMD()
        self.assertAlmostEqual(mmd(X, Y), mmd(Y, X))

    def test_emptyset(self):
        mmd = MMD()
        X = []
        Y = [[1, 2], [3, 4]]
        self.assertAlmostEqual(mmd(X, Y), mmd(Y, X))
        self.assertAlmostEqual(mmd(X, Y), 0.50915782)

    def test_cached_mmd_1(self):
        X, Y = [[1, 2], [3, 4]], [[5, 6]]
        mmd, cmmd = MMD(), CachedMMD(X)
        self.assertAlmostEqual(mmd(X, Y), cmmd(Y))
        self.assertAlmostEqual(mmd(X, X), cmmd(X))
        self.assertAlmostEqual(mmd([], X), cmmd([]))

    def test_cached_mmd_empty(self):
        mmd, cmmd = MMD(), CachedMMD([])
        self.assertAlmostEqual(mmd([[1, 2]], []), cmmd([[1, 2]]))

    def test_error_on_nonmatching_datasets(self):
        with self.assertRaises(ValueError):
            MMD()([[1, 2]], [[1]])

    def test_error_on_invalid_kernel(self):
        with self.assertRaises(ValueError):
            MMD(kernel="invalid")