"""Cross-closure regressions caught while splitting the Torch installer."""

from __future__ import print_function

import unittest

import numpy as np

import jittor as jt


class TestInstallerSplitRegressions(unittest.TestCase):
    def test_integer_getitem_remains_a_basic_index(self):
        value = jt.array([10, 20, 30])[1]
        self.assertEqual(int(value.item()), 20)

    def test_single_argument_where_uses_native_nonzero(self):
        indices = jt.where(jt.array([0, 1, 0, 1]).bool())
        self.assertIsInstance(indices, tuple)
        self.assertEqual(len(indices), 1)
        np.testing.assert_array_equal(
            indices[0].numpy().reshape(-1), np.array([1, 3])
        )

    def test_svd_returns_the_three_torch_values(self):
        u, singular_values, v = jt.svd(
            jt.array([[3.0, 0.0], [0.0, 2.0]])
        )
        self.assertEqual(tuple(u.shape), (2, 2))
        self.assertEqual(tuple(singular_values.shape), (2,))
        self.assertEqual(tuple(v.shape), (2, 2))


if __name__ == "__main__":
    unittest.main()
