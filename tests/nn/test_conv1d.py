"""Contracts for the ``nn.Conv1d`` module."""

import unittest

import numpy as np

import jittor as jt
from jittor import nn


def _reference(x, weight, bias):
    """Direct kernel_size=1 convolution: a per-position matrix multiply."""
    out = np.einsum("ncl,ock->nol", x, weight)
    if bias is not None:
        out = out + bias.reshape(1, -1, 1)
    return out


class TestConv1dParameterReplacement(unittest.TestCase):
    """A replaced parameter must be the one the layer convolves with.

    Checkpoint loaders rebind ``module._parameters[name]`` to a new Var rather than writing into
    the existing one, so a layer that captured its parameters at construction would keep using
    the initial values while reporting the loaded ones -- a wrong result with no error.
    """

    def setUp(self):
        rng = np.random.default_rng(0)
        self.x = rng.standard_normal((2, 3, 7), dtype=np.float32)
        self.weight = rng.standard_normal((5, 3, 1), dtype=np.float32)
        self.bias = rng.standard_normal(5, dtype=np.float32)

    def _check(self, use_cuda):
        with jt.flag_scope(use_cuda=use_cuda):
            layer = nn.Conv1d(3, 5, 1, bias=True)
            layer.weight = jt.array(self.weight)
            layer.bias = jt.array(self.bias)
            actual = layer(jt.array(self.x)).numpy()
        np.testing.assert_allclose(
            actual, _reference(self.x, self.weight, self.bias), atol=1e-5, rtol=1e-5
        )

    def test_replaced_parameters_are_used_cpu(self):
        self._check(use_cuda=0)

    @unittest.skipUnless(jt.has_cuda, "no CUDA device")
    def test_replaced_parameters_are_used_cuda(self):
        self._check(use_cuda=1)

    def test_assigned_parameters_are_used(self):
        """In-place assignment must keep working alongside replacement."""
        with jt.flag_scope(use_cuda=0):
            layer = nn.Conv1d(3, 5, 1, bias=True)
            layer.weight.assign(jt.array(self.weight))
            layer.bias.assign(jt.array(self.bias))
            actual = layer(jt.array(self.x)).numpy()
        np.testing.assert_allclose(
            actual, _reference(self.x, self.weight, self.bias), atol=1e-5, rtol=1e-5
        )

    def test_replaced_weight_without_bias(self):
        with jt.flag_scope(use_cuda=0):
            layer = nn.Conv1d(3, 5, 1, bias=False)
            layer.weight = jt.array(self.weight)
            actual = layer(jt.array(self.x)).numpy()
        np.testing.assert_allclose(
            actual, _reference(self.x, self.weight, None), atol=1e-5, rtol=1e-5
        )

    def test_replacement_takes_effect_after_a_forward(self):
        """A layer already used once must pick up parameters replaced afterwards."""
        with jt.flag_scope(use_cuda=0):
            layer = nn.Conv1d(3, 5, 1, bias=True)
            layer(jt.array(self.x))
            layer.weight = jt.array(self.weight)
            layer.bias = jt.array(self.bias)
            actual = layer(jt.array(self.x)).numpy()
        np.testing.assert_allclose(
            actual, _reference(self.x, self.weight, self.bias), atol=1e-5, rtol=1e-5
        )


if __name__ == "__main__":
    unittest.main()
