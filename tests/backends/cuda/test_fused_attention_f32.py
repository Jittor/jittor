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


def _reference(q, k, v, causal, dout, mask=None):
    q, k, v, dout = (t.astype(np.float64) for t in (q, k, v, dout))
    scale = 1 / np.sqrt(q.shape[-1])
    s = q @ np.swapaxes(k, -1, -2) * scale
    if causal:
        s = np.where(np.triu(np.ones(s.shape[-2:], bool), 1), -np.inf, s)
    if mask is not None:
        s = np.where(mask, s, -np.inf) if mask.dtype == bool else s + mask
    top = s.max(-1, keepdims=True)
    p = np.exp(s - np.where(np.isinf(top), 0, top))
    total = p.sum(-1, keepdims=True)
    # A row with every key masked answers 0, as the composite path does.
    p = np.divide(p, total, out=np.zeros_like(p), where=total > 0)
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

    def _check(self, b, h, lq, lk, d, causal, mask=None):
        rng = np.random.RandomState(d + lq)
        q, dout = (rng.randn(b, h, lq, d).astype("float32") for _ in range(2))
        k, v = (rng.randn(b, h, lk, d).astype("float32") for _ in range(2))
        want = _reference(q, k, v, causal, dout, mask)
        calls = []
        original = self.kernel._forward

        def counted(*args):
            calls.append(1)
            return original(*args)

        self.kernel._forward = counted
        self.addCleanup(setattr, self.kernel, "_forward", original)
        jq, jk, jv = (jt.array(t) for t in (q, k, v))
        out = scaled_dot_product_attention(
            jq, jk, jv, attn_mask=None if mask is None else jt.array(mask).stop_grad(),
            is_causal=causal)
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

    # Masks. Transformers passes an explicit one whenever
    # `torch.compiler.is_compiling()`, a padded batch always does, and both
    # used to decline to the path that writes every score out.
    def test_a_key_padding_mask(self):
        mask = np.ones((2, 1, 1, 47), bool)
        mask[0, ..., 30:] = False
        mask[1, ..., :5] = False
        self._check(2, 3, 33, 47, 40, False, mask)

    def test_an_additive_mask_per_head(self):
        rng = np.random.RandomState(7)
        self._check(2, 2, 70, 45, 64, False,
                    rng.randn(2, 2, 70, 45).astype("float32"))

    def test_an_explicit_causal_mask_on_wide_heads(self):
        # What Transformers builds: [batch, 1, q, k], True where visible.
        mask = np.broadcast_to(np.tril(np.ones((96, 96), bool)), (2, 1, 96, 96)).copy()
        self._check(2, 2, 96, 96, 128, False, mask)

    def test_a_mask_on_top_of_causal_and_rows_masked_whole(self):
        mask = np.ones((40, 40), bool)
        mask[5] = False
        mask[:, 0] = False
        self._check(1, 2, 40, 40, 32, True, mask)

    def test_a_small_inference_takes_the_narrow_query_tiles(self):
        # A grid smaller than the device takes one query row a thread: BERT at
        # batch 1 is 24 tiles of 64 queries otherwise.
        rng = np.random.RandomState(4)
        q, k, v = (rng.randn(1, 2, 16, 32).astype("float32") for _ in range(3))
        with jt.no_grad():
            out = self.kernel._fused_attention_f32(*(jt.array(t) for t in (q, k, v)))
        self.assertIsNotNone(out)
        want = _reference(q, k, v, False, np.zeros_like(q))[0]
        np.testing.assert_allclose(out.numpy(), want, rtol=1e-4, atol=1e-5)

    def test_what_it_declines(self):
        q = jt.random((1, 2, 8, 256))
        self.assertIsNone(self.kernel._fused_attention_f32(q, q, q))
        q = jt.random((1, 2, 8, 64))
        self.assertIsNone(self.kernel._fused_attention_f32(q, q, q, dropout_p=0.1))
        # A learned bias needs its own gradient; a half-precision one is not
        # what this float32 kernel reads; a mask must broadcast.
        self.assertIsNone(self.kernel._fused_attention_f32(
            q, q, q, attn_mask=jt.zeros((8, 8))))
        self.assertIsNone(self.kernel._fused_attention_f32(
            q, q, q, attn_mask=jt.zeros((8, 8)).float16().stop_grad()))
        self.assertIsNone(self.kernel._fused_attention_f32(
            q, q, q, attn_mask=jt.ones((3, 8), dtype="bool")))

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
