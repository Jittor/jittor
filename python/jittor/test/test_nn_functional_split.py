"""Independent numeric coverage for functions moved out of ``jittor.nn``."""

import unittest

import numpy as np

import jittor as jt
from jittor import nn


class TestNNFunctionalSplit(unittest.TestCase):
    def _check_rrelu(self, use_cuda):
        values = np.linspace(-4.0, 4.0, 1025, dtype=np.float32)
        lower, upper = 0.1, 0.4
        with jt.flag_scope(use_cuda=int(use_cuda)):
            x = jt.array(values)
            evaluated = nn.rrelu(x, lower=lower, upper=upper, training=False).numpy()
            expected = np.where(values >= 0, values, values * ((lower + upper) / 2))
            np.testing.assert_allclose(evaluated, expected, rtol=1e-6, atol=1e-6)

            trained = nn.rrelu(x, lower=lower, upper=upper, training=True).numpy()
            negative = values < 0
            slopes = trained[negative] / values[negative]
            self.assertGreaterEqual(float(slopes.min()), lower - 1e-6)
            self.assertLessEqual(float(slopes.max()), upper + 1e-6)
            np.testing.assert_allclose(trained[~negative], values[~negative], atol=0)

    def _check_pairwise_distance(self, use_cuda):
        x1 = np.array(
            [[1.5, -2.0, 0.25], [-3.0, 0.5, 4.0]], dtype=np.float32
        )
        x2 = np.array(
            [[-0.5, 1.0, 0.75], [2.0, -1.5, 1.0]], dtype=np.float32
        )
        eps = 1e-4
        with jt.flag_scope(use_cuda=int(use_cuda)):
            a, b = jt.array(x1), jt.array(x2)
            for p in (1.0, 2.0, 3.0, float("inf")):
                for keepdim in (False, True):
                    with self.subTest(use_cuda=use_cuda, p=p, keepdim=keepdim):
                        actual = nn.pairwise_distance(
                            a, b, p=p, eps=eps, keepdim=keepdim
                        ).numpy()
                        diff = np.abs(x1 - x2 + eps)
                        expected = (
                            np.max(diff, axis=-1, keepdims=keepdim)
                            if np.isinf(p)
                            else np.sum(diff ** p, axis=-1, keepdims=keepdim) ** (1.0 / p)
                        )
                        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)

    def test_rrelu_cpu(self):
        self._check_rrelu(use_cuda=False)

    def test_pairwise_distance_cpu(self):
        self._check_pairwise_distance(use_cuda=False)

    @unittest.skipIf(not jt.has_cuda, "CUDA is not available")
    def test_rrelu_cuda(self):
        self._check_rrelu(use_cuda=True)

    @unittest.skipIf(not jt.has_cuda, "CUDA is not available")
    def test_pairwise_distance_cuda(self):
        self._check_pairwise_distance(use_cuda=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
