# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""The memory-efficient float32 attention kernel, against the closed form.

Built from matmuls, float32 attention with its backward held 4 GiB and took
25 ms for a 4096-token SD1.5 call; this kernel holds 110 MiB and takes 10 ms,
because it never writes the score matrix out. The shapes below cover what its
tiling can get wrong: partial query and key tiles, lengths that differ, head
dimensions on and off the tile widths, and the causal mask's skipped tiles.
The reference is float64 numpy.
"""

from _helpers import capability as _test_capability
import unittest

import numpy as np

import jittor as jt
from jittor.nn.functional.attention import scaled_dot_product_attention


def _reference(q, k, v, causal, dout):
    q, k, v, dout = (t.astype(np.float64) for t in (q, k, v, dout))
    scale = 1 / np.sqrt(q.shape[-1])
    s = q @ np.swapaxes(k, -1, -2) * scale
    if causal:
        s = np.where(np.triu(np.ones(s.shape[-2:], bool), 1), -np.inf, s)
    p = np.exp(s - s.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    dv = np.swapaxes(p, -1, -2) @ dout
    dp = dout @ np.swapaxes(v, -1, -2)
    ds = p * (dp - (dp * p).sum(-1, keepdims=True))
    return p @ v, ds @ k * scale, np.swapaxes(ds, -1, -2) @ q * scale, dv


@unittest.skipIf(not _test_capability.check_accelerator("cuda", backend=jt).enabled,
                 "no usable CUDA in this build")
class TestFusedAttentionF32(unittest.TestCase):
    def setUp(self):
        from jittor.backends.cuda.kernels.nn import fused_attention_f32_cuda
        self.kernel = fused_attention_f32_cuda
        self.scope = jt.flag_scope(use_cuda=1)
        self.scope.__enter__()
        self.addCleanup(self.scope.__exit__, None, None, None)

    def _check(self, b, h, lq, lk, d, causal):
        rng = np.random.RandomState(d + lq)
        q, dout = (rng.randn(b, h, lq, d).astype("float32") for _ in range(2))
        k, v = (rng.randn(b, h, lk, d).astype("float32") for _ in range(2))
        want = _reference(q, k, v, causal, dout)
        calls = []
        original = self.kernel._forward

        def counted(*args):
            calls.append(1)
            return original(*args)

        self.kernel._forward = counted
        self.addCleanup(setattr, self.kernel, "_forward", original)
        jq, jk, jv = (jt.array(t) for t in (q, k, v))
        out = scaled_dot_product_attention(jq, jk, jv, is_causal=causal)
        grads = jt.grad((out * jt.array(dout)).sum(), [jq, jk, jv])
        self.assertEqual(calls, [1])
        for name, got, expected in zip(("out", "dq", "dk", "dv"), [out] + list(grads), want):
            np.testing.assert_allclose(got.numpy(), expected, rtol=1e-4,
                                       atol=1e-5 * np.abs(expected).max(), err_msg=name)

    def test_partial_tiles_and_unequal_lengths(self):
        self._check(2, 3, 33, 47, 40, False)

    def test_causal_across_several_tiles(self):
        self._check(1, 2, 130, 130, 64, True)

    def test_wide_heads_take_the_narrow_tiles(self):
        self._check(2, 2, 100, 37, 128, False)

    def test_a_head_dimension_off_the_lane_width(self):
        self._check(1, 1, 129, 129, 80, True)

    def test_tiny(self):
        self._check(2, 2, 5, 3, 8, False)

    def test_what_it_declines(self):
        q = jt.random((1, 2, 8, 256))
        self.assertIsNone(self.kernel._fused_attention_f32(q, q, q))
        q = jt.random((1, 2, 8, 64))
        self.assertIsNone(self.kernel._fused_attention_f32(
            q, q, q, attn_mask=jt.ones((8, 8), dtype="bool")))
        self.assertIsNone(self.kernel._fused_attention_f32(q, q, q, dropout_p=0.1))

    def test_the_scores_are_never_built(self):
        heads, tokens, width = 8, 4096, 40
        scores = 2 * heads * tokens * tokens * 4          # 1 GiB
        q, k, v = (jt.random((2, heads, tokens, width)) for _ in range(3))
        for t in (q, k, v):
            t.sync()
        jt.sync_all(True)
        base = jt.core.device_memory_used(0)
        jt.core.reset_device_memory_peak(0)
        out = scaled_dot_product_attention(q, k, v)
        jt.sync(jt.grad(out.sum(), [q, k, v]))
        jt.sync_all(True)
        peak = jt.core.device_memory_peak(0) - base
        self.assertLess(peak, scores / 8, "peaked at %.0f MiB" % (peak / 2**20))


if __name__ == "__main__":
    unittest.main()
