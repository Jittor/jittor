"""Dtype contract for the torch-compatible ``nn.RMSNorm``."""

import unittest

import numpy as np

import jittor as jt


def _reference(x, weight, eps):
    """RMSNorm in float64: scale by the root mean square over the last axis, then apply weight."""
    x64 = x.astype(np.float64)
    scale = 1.0 / np.sqrt((x64 * x64).mean(-1, keepdims=True) + eps)
    return x64 * scale * weight.astype(np.float64)


class TestRMSNormDtype(unittest.TestCase):
    """RMSNorm must answer in the dtype it was given.

    The mean square is accumulated in float32 for range, but torch returns the input's dtype even
    when the weight is wider. Widening the result instead would spread float32 through everything
    downstream -- in a half-precision transformer, that is the whole attention path, which then
    costs twice the memory and can no longer reach fused attention kernels.
    """

    eps = 1e-5

    def setUp(self):
        from torch import nn  # the shim's nn, installed onto jittor

        rng = np.random.default_rng(0)
        self.x = (rng.standard_normal((3, 8, 64)) * 3.0).astype(np.float32)
        self.weight = (rng.standard_normal(64) * 0.2 + 1.0).astype(np.float32)
        self.nn = nn

    def _run(self, dtype, use_cuda, tolerance):
        with jt.flag_scope(use_cuda=use_cuda):
            layer = self.nn.RMSNorm(64, eps=self.eps)
            layer.weight.assign(jt.array(self.weight))
            x = jt.array(self.x).cast(dtype)
            out = layer(x)
            self.assertEqual(str(out.dtype), dtype)
            actual = out.float32().numpy()
            rounded = x.float32().numpy()
        expected = _reference(rounded, self.weight, self.eps)
        np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=tolerance)

    def test_float32_cpu(self):
        self._run("float32", use_cuda=0, tolerance=1e-5)

    def test_float16_cpu(self):
        self._run("float16", use_cuda=0, tolerance=6e-3)

    @unittest.skipUnless(jt.has_cuda, "no CUDA device")
    def test_float32_cuda(self):
        self._run("float32", use_cuda=1, tolerance=1e-5)

    @unittest.skipUnless(jt.has_cuda, "no CUDA device")
    def test_bfloat16_cuda(self):
        self._run("bfloat16", use_cuda=1, tolerance=6e-2)

    @unittest.skipUnless(jt.has_cuda, "no CUDA device")
    def test_float16_cuda(self):
        self._run("float16", use_cuda=1, tolerance=6e-3)

    def test_without_weight_keeps_dtype(self):
        with jt.flag_scope(use_cuda=0):
            layer = self.nn.RMSNorm(64, eps=self.eps, elementwise_affine=False)
            out = layer(jt.array(self.x).cast("float16"))
            self.assertEqual(str(out.dtype), "float16")


if __name__ == "__main__":
    unittest.main()
