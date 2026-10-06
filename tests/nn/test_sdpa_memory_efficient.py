# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Attention that trains keeps its inputs and output, not the probabilities.

The math-path `scaled_dot_product_attention` kept the ``[..., Lq, Lk]``
softmax (and, through autodiff, the scores) for its backward. At the SD1.5
UNet's 64x64 resolution one call retained 522 MB after its forward where
PyTorch's memory-efficient kernel retains 5 MB, and an SD1.5 UNet training
step ran out of a 24 GB card. With gradients required the forward now keeps
q, k, v and the output, and the backward recomputes the probabilities.

Attention without a backward keeps nothing, but the composite still builds
the whole score matrix and its softmax at once: 1 GiB per call at the same
resolution, where PyTorch's fused kernel holds none, and SD1.5 sampling peaked
0.9 GB above PyTorch. Above a size it now runs in blocks of queries.

The reference is the closed form in float64 numpy -- softmax attention and
its gradients -- including rows every key is masked out of, which produce 0
and receive zero gradient as the fast softmax's ``zero_all_neg_inf`` does.
"""

from _helpers import capability as _test_capability

import unittest

import numpy as np

import jittor as jt
from jittor.nn.functional import attention
from jittor.nn.functional.attention import scaled_dot_product_attention


def _has_cuda():
    return bool(_test_capability.check_accelerator("cuda", backend=jt).enabled)


def _devices():
    # Asked when a test runs, not at collection: collection must not probe
    # the backend (tests/structure/test_pytest_contract.py).
    return [("cpu", 0)] + ([("cuda", 1)] if _has_cuda() else [])


def _reference(q, k, v, mask, causal, dout):
    q, k, v, dout = (t.astype(np.float64) for t in (q, k, v, dout))
    scale = 1 / np.sqrt(q.shape[-1])
    s = q @ np.swapaxes(k, -1, -2) * scale
    if causal:
        s = np.where(np.triu(np.ones(s.shape[-2:], bool), 1), -np.inf, s)
    if mask is not None:
        s = np.where(mask, s, -np.inf) if mask.dtype == bool else s + mask
    top = s.max(-1, keepdims=True)
    top = np.where(np.isfinite(top), top, 0)
    e = np.exp(s - top)
    total = e.sum(-1, keepdims=True)
    p = np.where(total > 0, e / np.where(total > 0, total, 1), 0)
    out = p @ v
    dv = np.swapaxes(p, -1, -2) @ dout
    dp = dout @ np.swapaxes(v, -1, -2)
    ds = p * (dp - (dp * p).sum(-1, keepdims=True))
    return out, ds @ k * scale, np.swapaxes(ds, -1, -2) @ q * scale, dv


class TestTrainingAttentionMatchesTheClosedForm(unittest.TestCase):
    def _check(self, lq, lk, mask=None, causal=False):
        rng = np.random.RandomState(0)
        b, h, d = 2, 3, 8
        q = rng.randn(b, h, lq, d).astype("float32")
        k = rng.randn(b, h, lk, d).astype("float32")
        v = rng.randn(b, h, lk, d).astype("float32")
        dout = rng.randn(b, h, lq, d).astype("float32")
        want = _reference(q, k, v, mask, causal, dout)
        for device, use_cuda in _devices():
            with self.subTest(device=device):
                with jt.flag_scope(use_cuda=use_cuda):
                    jq, jk, jv = jt.array(q), jt.array(k), jt.array(v)
                    out = scaled_dot_product_attention(
                        jq, jk, jv, is_causal=causal,
                        attn_mask=None if mask is None else jt.array(mask))
                    grads = jt.grad((out * jt.array(dout)).sum(), [jq, jk, jv])
                    for got, expected in zip([out] + list(grads), want):
                        np.testing.assert_allclose(got.numpy(), expected,
                                                   rtol=1e-5, atol=2e-6)

    def test_no_mask_and_unequal_lengths(self):
        self._check(7, 9)

    def test_bool_mask_with_a_fully_masked_row(self):
        mask = np.random.RandomState(1).rand(2, 1, 7, 9) > 0.3
        mask[0, 0, 2, :] = False
        self._check(7, 9, mask=mask)

    def test_additive_float_mask(self):
        mask = (np.random.RandomState(2).randn(2, 3, 7, 9) * 0.5).astype("float32")
        self._check(7, 9, mask=mask)

    def test_causal(self):
        self._check(9, 9, causal=True)


class TestOnlyTrainingTakesTheFunction(unittest.TestCase):
    def test_inference_and_dropout_keep_the_composite(self):
        # Nothing to save without a backward, and dropout needs the composite's
        # mask; both stay on the path they were on.
        seen = []
        original = attention._MemoryEfficientAttention.apply
        attention._MemoryEfficientAttention.apply = classmethod(
            lambda cls, *a: seen.append(1) or original(*a))
        try:
            q, k, v = (jt.random((1, 2, 16, 8)) for _ in range(3))
            with jt.no_grad():
                scaled_dot_product_attention(q, k, v).sync()
            scaled_dot_product_attention(q, k, v, dropout_p=0.5).sync()
            self.assertEqual(seen, [])
            scaled_dot_product_attention(q, k, v).sync()
            self.assertEqual(seen, [1])
        finally:
            # The patch shadows the classmethod inherited from jt.Function.
            del attention._MemoryEfficientAttention.apply


class TestInferenceAttentionInBlocks(unittest.TestCase):
    def setUp(self):
        # Three queries per block for the shapes below: 7 queries split 3+3+1,
        # so a block boundary and a short last block are both exercised.
        original = attention._SCORE_CHUNK_BYTES
        attention._SCORE_CHUNK_BYTES = 2 * 3 * 9 * 4 * 3
        self.addCleanup(setattr, attention, "_SCORE_CHUNK_BYTES", original)

    def _check(self, mask=None):
        rng = np.random.RandomState(3)
        q = rng.randn(2, 3, 7, 8).astype("float32")
        k = rng.randn(2, 3, 9, 8).astype("float32")
        v = rng.randn(2, 3, 9, 8).astype("float32")
        want = _reference(q, k, v, mask, False, np.zeros_like(q))[0]
        self.assertEqual(attention._query_block(jt.array(q), jt.array(k), False), 3)
        for device, use_cuda in _devices():
            with self.subTest(device=device), jt.flag_scope(use_cuda=use_cuda), jt.no_grad():
                got = scaled_dot_product_attention(
                    jt.array(q), jt.array(k), jt.array(v),
                    attn_mask=None if mask is None else jt.array(mask))
                np.testing.assert_allclose(got.numpy(), want, rtol=1e-5, atol=2e-6)

    def test_no_mask(self):
        self._check()

    def test_a_mask_per_query_is_sliced_with_the_queries(self):
        mask = np.random.RandomState(4).rand(2, 1, 7, 9) > 0.3
        mask[1, 0, 4, :] = False
        self._check(mask)

    def test_a_two_dimensional_float_mask(self):
        self._check((np.random.RandomState(5).randn(7, 9) * 0.5).astype("float32"))

    def test_a_mask_broadcast_over_queries_is_passed_whole(self):
        self._check(np.random.RandomState(6).rand(2, 1, 1, 9) > 0.2)


@unittest.skipIf(not _has_cuda(), "the pools count device memory only")
class TestTrainingAttentionKeepsNoScores(unittest.TestCase):
    def test_the_forward_retains_far_less_than_one_score_matrix(self):
        tokens, heads, width = 2048, 8, 40
        scores = heads * tokens * tokens * 4          # 128 MiB
        with jt.flag_scope(use_cuda=1):
            q, k, v = (jt.random((1, heads, tokens, width)) for _ in range(3))
            jt.sync_all(True)
            before = jt.core.device_memory_used(0)
            out = scaled_dot_product_attention(q, k, v)
            out.sync()
            jt.sync_all(True)
            retained = jt.core.device_memory_used(0) - before
            (dq,) = jt.grad(out.sum(), [q])
            dq.sync()
        # The output itself is heads*tokens*width floats (2.5 MiB); the
        # probabilities the composite kept were the whole 128 MiB.
        self.assertLess(retained, scores / 8,
                        "the forward retained %.0f MiB" % (retained / 2**20))



@unittest.skipIf(not _has_cuda(), "the pools count device memory only")
class TestInferenceAttentionBuildsNoFullScores(unittest.TestCase):
    def test_the_peak_is_a_fraction_of_one_score_matrix(self):
        heads, tokens, width = 8, 4096, 40
        scores = 2 * heads * tokens * tokens * 2       # float16, 512 MiB
        with jt.flag_scope(use_cuda=1), jt.no_grad():
            q, k, v = (jt.random((2, heads, tokens, width)).float16() for _ in range(3))
            jt.sync_all(True)
            base = jt.core.device_memory_used(0)
            jt.core.reset_device_memory_peak(0)
            scaled_dot_product_attention(q, k, v).sync()
            jt.sync_all(True)
            peak = jt.core.device_memory_peak(0) - base
        # Built at once, the scores and their softmax were 2x this.
        self.assertLess(peak, scores, "peaked at %.0f MiB" % (peak / 2**20))


if __name__ == "__main__":
    unittest.main()
