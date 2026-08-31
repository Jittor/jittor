"""Native reset-parameter contracts shared with Torch-facing runtimes."""

import unittest

import numpy as np

import jittor as jt
from jittor import nn


class TestResetParameters(unittest.TestCase):
    def test_linear_reset_preserves_parameter_identity(self):
        with jt.flag_scope(use_cuda=0):
            module = nn.Linear(8, 4)
            weight = module.weight
            bias = module.bias
            weight.update(jt.zeros_like(weight))
            bias.update(jt.zeros_like(bias))
            module.reset_parameters()
            actual_weight = weight.numpy()
            actual_bias = bias.numpy()

        self.assertIs(module.weight, weight)
        self.assertIs(module.bias, bias)
        self.assertTrue(np.isfinite(actual_weight).all())
        self.assertTrue(np.isfinite(actual_bias).all())
        self.assertGreater(float(np.abs(actual_weight).max()), 0.0)
        self.assertGreater(float(np.abs(actual_bias).max()), 0.0)

    def test_embedding_reset_clears_padding_row(self):
        with jt.flag_scope(use_cuda=0):
            module = nn.Embedding(6, 3, padding_idx=2)
            weight = module.weight
            weight.update(jt.ones_like(weight) * 7.0)
            module.reset_parameters()
            actual = weight.numpy()

        self.assertIs(module.weight, weight)
        np.testing.assert_array_equal(actual[2], np.zeros((3,), dtype=np.float32))
        self.assertGreater(float(np.abs(np.delete(actual, 2, axis=0)).max()), 0.0)

    def test_layer_norm_reset_restores_affine_defaults(self):
        with jt.flag_scope(use_cuda=0):
            module = nn.LayerNorm(5)
            weight = module.weight
            bias = module.bias
            weight.update(jt.zeros_like(weight))
            bias.update(jt.ones_like(bias))
            module.reset_parameters()
            actual_weight = weight.numpy()
            actual_bias = bias.numpy()

        self.assertIs(module.weight, weight)
        self.assertIs(module.bias, bias)
        np.testing.assert_array_equal(actual_weight, np.ones((5,), dtype=np.float32))
        np.testing.assert_array_equal(actual_bias, np.zeros((5,), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
