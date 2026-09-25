# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""cuDNN's fused attention, checked against the closed form.

The math path of ``scaled_dot_product_attention`` writes the whole score
matrix and its softmax to device memory: 1 GiB and 2.9 ms per call at the
SD1.5 UNet's 64x64 resolution, where this kernel takes 20 MiB and 0.5 ms.
It serves float16 and bfloat16, forward and backward, causal masks, and --
without a backward -- bool and float masks. The
reference is float64 numpy, with a row every key is masked out of giving 0 as
the composite does.
"""

from _helpers import capability as _test_capability
import importlib.util
import unittest

import numpy as np

import jittor as jt
from jittor.nn.functional.attention import scaled_dot_product_attention


def _usable():
    return (bool(_test_capability.check_accelerator("cuda", backend=jt).enabled)
            and importlib.util.find_spec("cudnn") is not None)


def _reference(q, k, v, mask, causal, dout):
    q, k, v, dout = (t.astype(np.float64) for t in (q, k, v, dout))
    groups = q.shape[1] // k.shape[1]
    k, v = np.repeat(k, groups, 1), np.repeat(v, groups, 1)
    scale = 1 / np.sqrt(q.shape[-1])
    s = q @ np.swapaxes(k, -1, -2) * scale
    if causal:
        s = np.where(np.triu(np.ones(s.shape[-2:], bool), 1), -np.inf, s)
    if mask is not None:
        s = np.where(mask, s, -np.inf) if mask.dtype == bool else s + mask
    top = s.max(-1, keepdims=True)
    e = np.exp(s - np.where(np.isfinite(top), top, 0))
    total = e.sum(-1, keepdims=True)
    p = np.where(total > 0, e / np.where(total > 0, total, 1), 0)
    dv = np.swapaxes(p, -1, -2) @ dout
    dp = dout @ np.swapaxes(v, -1, -2)
    ds = p * (dp - (dp * p).sum(-1, keepdims=True))

    def fold(g):
        return g.reshape(g.shape[0], -1, groups, *g.shape[2:]).sum(2)
    return p @ v, ds @ k * scale, fold(np.swapaxes(ds, -1, -2) @ q * scale), fold(dv)


@unittest.skipIf(not _usable(), "needs CUDA and the nvidia-cudnn-frontend headers")
class TestCudnnFusedAttention(unittest.TestCase):
    def setUp(self):
        from jittor.backends.cuda.kernels.nn import cudnn_attention_cuda
        self.kernel = cudnn_attention_cuda
        self.kernel._supported_shapes.clear()
        self.scope = jt.flag_scope(use_cuda=1)
        self.scope.__enter__()
        self.addCleanup(self.scope.__exit__, None, None, None)

    def _ran_on_cudnn(self):
        # The kernel asks cuDNN once per shape; an answer of True is a call
        # it then served.
        return any(self.kernel._supported_shapes.values())

    def _check_training(self, dtype, b, h, hk, sq, skv, d, causal, tolerance):
        rng = np.random.RandomState(0)
        q, dout = (rng.randn(b, h, sq, d).astype("float32") for _ in range(2))
        k, v = (rng.randn(b, hk, skv, d).astype("float32") for _ in range(2))
        want = _reference(q, k, v, None, causal, dout)
        jq, jk, jv = (jt.array(t).cast(dtype) for t in (q, k, v))
        out = scaled_dot_product_attention(jq, jk, jv, is_causal=causal)
        grads = jt.grad((out.float32() * jt.array(dout)).sum(), [jq, jk, jv])
        self.assertTrue(self._ran_on_cudnn())
        for name, got, expected in zip(("out", "dq", "dk", "dv"), [out] + list(grads), want):
            error = np.abs(got.float32().numpy() - expected).max() / np.abs(expected).max()
            self.assertLess(error, tolerance, name)

    def test_float16_unequal_lengths(self):
        self._check_training("float16", 2, 4, 4, 33, 47, 64, False, 3e-3)

    def test_float16_causal(self):
        self._check_training("float16", 1, 8, 8, 64, 64, 40, True, 3e-3)

    def test_unequal_head_counts_are_left_to_the_other_path(self):
        # There is no enable_gqa here; the mismatch is the caller's error.
        q = jt.random((1, 4, 8, 64)).float16()
        k = jt.random((1, 2, 8, 64)).float16()
        self.assertIsNone(self.kernel._cudnn_fused_attention(q, k, k))

    def test_bfloat16(self):
        self._check_training("bfloat16", 2, 2, 2, 128, 128, 128, False, 2e-2)

    def _check_mask(self, mask):
        rng = np.random.RandomState(1)
        q = rng.randn(2, 4, 48, 64).astype("float32")
        k, v = (rng.randn(2, 4, 64, 64).astype("float32") for _ in range(2))
        want = _reference(q, k, v, mask, False, np.zeros_like(q))[0]
        with jt.no_grad():
            got = scaled_dot_product_attention(
                *(jt.array(t).float16() for t in (q, k, v)),
                attn_mask=jt.array(mask) if mask.dtype == bool else jt.array(mask).float16())
            got = got.float32().numpy()
        self.assertTrue(self._ran_on_cudnn())
        np.testing.assert_allclose(got, want, atol=3e-3 * np.abs(want).max())
        return got

    def test_a_bool_mask_and_a_row_with_no_keys(self):
        mask = np.random.RandomState(2).rand(2, 1, 48, 64) > 0.3
        mask[1, 0, 5, :] = False
        got = self._check_mask(mask)
        self.assertEqual(np.abs(got[1, :, 5]).max(), 0)

    def test_a_float_mask_per_head(self):
        self._check_mask((np.random.RandomState(3).randn(2, 4, 48, 64) * 0.5).astype("float32"))

    def test_a_two_dimensional_mask(self):
        self._check_mask((np.random.RandomState(4).randn(48, 64) * 0.5).astype("float32"))

    def test_training_with_a_mask_keeps_the_other_path(self):
        q = jt.random((1, 2, 16, 64)).float16()
        mask = jt.ones((16, 16), dtype="bool")
        self.assertIsNone(self.kernel._cudnn_fused_attention(q, q, q, attn_mask=mask))

    def test_the_scores_are_never_built(self):
        heads, tokens, width = 8, 4096, 40
        scores = 2 * heads * tokens * tokens * 2          # float16, 512 MiB
        q, k, v = (jt.random((2, heads, tokens, width)).float16() for _ in range(3))
        for t in (q, k, v):
            t.sync()
        for grad in (False, True):
            with self.subTest(training=grad):
                jt.sync_all(True)
                base = jt.core.device_memory_used(0)
                jt.core.reset_device_memory_peak(0)
                if grad:
                    out = scaled_dot_product_attention(q, k, v)
                    jt.sync(jt.grad(out.float32().sum(), [q, k, v]))
                else:
                    with jt.no_grad():
                        scaled_dot_product_attention(q, k, v).sync()
                jt.sync_all(True)
                peak = jt.core.device_memory_peak(0) - base
                self.assertLess(peak, scores / 4, "peaked at %.0f MiB" % (peak / 2**20))


if __name__ == "__main__":
    unittest.main()
