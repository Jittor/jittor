# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""cuDNN's fused attention, checked against the closed form.

The math path of ``scaled_dot_product_attention`` writes the whole score
matrix and its softmax to device memory: 1 GiB and 2.9 ms per call at the
SD1.5 UNet's 64x64 resolution, where this kernel takes 20 MiB and 0.5 ms.
It serves float16 and bfloat16, forward and backward, causal masks, and constant bool/float masks. The
reference is float64 numpy, with a row every key is masked out of giving 0 as
the composite does.
"""

from _helpers import capability as _test_capability
import importlib.util
import unittest

import numpy as np

import jittor as jt
from jittor.nn.functional.attention import LAST_FUSED_KERNEL, scaled_dot_product_attention
from jittor._runtime import forbid_backend_fallbacks


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
        # A masked call leaves its mask's bias cached for the next layer.
        self.addCleanup(self.kernel._MASK_TERMS.clear)
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

    def _check_seq_major(self, which, mask=None):
        # q/k/v as `x.view(b, s, h, d).transpose(1, 2)`, the way attention
        # blocks build them. The kernel reads the [b, s, h, d] tensors in
        # place and hands the output back as a transpose view, which the
        # caller's `transpose(1, 2)` then undoes without a copy.
        b, h, s, d = 2, 4, 70, 64
        rng = np.random.RandomState(1)
        q, k, v, dout = (rng.randn(b, h, s, d).astype("float32") for _ in range(4))
        want = _reference(q, k, v, mask, False, dout)
        sources = [jt.array(np.ascontiguousarray(t.transpose(0, 2, 1, 3))).float16()
                   if name in which else jt.array(t).float16()
                   for name, t in zip("qkv", (q, k, v))]
        heads = [t.transpose(1, 2) if name in which else t
                 for name, t in zip("qkv", sources)]
        attn_mask = None if mask is None else jt.array(mask)
        # A mask is served without a backward only.
        with jt.flag_scope(no_grad=mask is not None):
            out = scaled_dot_product_attention(*heads, attn_mask=attn_mask)
        seq_out = out.transpose(1, 2)
        self.assertTrue(any(key[-1] for key, ok in self.kernel._supported_shapes.items() if ok))
        np.testing.assert_allclose(seq_out.float32().numpy(), want[0].transpose(0, 2, 1, 3),
                                   atol=2e-3, rtol=2e-3)
        if mask is not None:
            return
        grads = jt.grad((out.float32() * jt.array(dout)).sum(), sources)
        for name, got, expected in zip("qkv", grads, want[1:]):
            if name in which:
                expected = expected.transpose(0, 2, 1, 3)
            error = np.abs(got.float32().numpy() - expected).max() / np.abs(expected).max()
            self.assertLess(error, 5e-3, "d" + name)

    def test_seq_major_inputs_are_read_in_place(self):
        self._check_seq_major("qkv")

    def test_a_seq_major_query_with_dense_keys(self):
        self._check_seq_major("q")

    def test_seq_major_inputs_with_a_mask(self):
        mask = np.ones((2, 1, 70, 70), bool)
        mask[1, ..., 50:] = False
        self._check_seq_major("qkv", mask)

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

    def test_one_mask_is_turned_into_a_bias_once(self):
        # A decoder gives every layer the same mask, which Transformers slices
        # afresh per layer: a new view of the same data. The bias is built for
        # the first and reused by the rest; a different mask is not confused
        # with it.
        rng = np.random.RandomState(3)
        q = rng.randn(1, 4, 48, 64).astype("float32")
        k, v = (rng.randn(1, 4, 64, 64).astype("float32") for _ in range(2))
        full = rng.rand(1, 1, 48, 80) > 0.3
        other = rng.rand(1, 1, 48, 64) > 0.5
        base = jt.array(full)
        self.kernel._MASK_TERMS.clear()
        with jt.no_grad():
            qkv = [jt.array(t).float16() for t in (q, k, v)]
            first = scaled_dot_product_attention(*qkv, attn_mask=base[:, :, :, :64])
            again = scaled_dot_product_attention(*qkv, attn_mask=base[:, :, :, :64])
            self.assertEqual(len(self.kernel._MASK_TERMS), 1)
            self.assertIs(self.kernel._MASK_TERMS[0][2],
                          self.kernel._mask_terms(base[:, :, :, :64], (1, 4, 48, 64),
                                                  (1, 4, 64, 64), qkv[0].dtype)[0])
            third = scaled_dot_product_attention(*qkv, attn_mask=jt.array(other))
            self.assertEqual(len(self.kernel._MASK_TERMS), 1)
        want = _reference(q, k, v, full[..., :64], False, np.zeros_like(q))[0]
        for got in (first, again):
            np.testing.assert_allclose(got.float32().numpy(), want, atol=3e-3 * np.abs(want).max())
        want = _reference(q, k, v, other, False, np.zeros_like(q))[0]
        np.testing.assert_allclose(third.float32().numpy(), want, atol=3e-3 * np.abs(want).max())

    def test_a_float_mask_per_head(self):
        self._check_mask((np.random.RandomState(3).randn(2, 4, 48, 64) * 0.5).astype("float32"))

    def test_a_two_dimensional_mask(self):
        self._check_mask((np.random.RandomState(4).randn(48, 64) * 0.5).astype("float32"))

    def _check_mask_training(self, dtype, mask, causal=False, grouped=False, expect_fused=True):
        rng = np.random.RandomState(41)
        heads, key_heads = (4, 2) if grouped else (2, 2)
        q = rng.randn(1, heads, 33, 64).astype("float32") * 0.5
        k, v = (rng.randn(1, key_heads, 47, 64).astype("float32") * 0.5 for _ in range(2))
        dout = rng.randn(*q.shape).astype("float32") * 0.5
        with forbid_backend_fallbacks():
            qkv = [jt.array(t).cast(dtype) for t in (q, k, v)]
            upstream = jt.array(dout).cast(dtype).stop_grad()
            # Compare independent float64 arithmetic on the actual quantized
            # operand values. The oracle never calls another Jittor attention.
            values = [t.float32().numpy() for t in qkv]
            want = _reference(*values, mask, causal, upstream.float32().numpy())
            LAST_FUSED_KERNEL[0] = None
            function = (self.kernel._cudnn_fused_attention_gqa if grouped
                        else scaled_dot_product_attention)
            out = function(*qkv, attn_mask=jt.array(mask).stop_grad(), is_causal=causal)
            self.assertIsNotNone(out)
            route = function if grouped else LAST_FUSED_KERNEL[0]
            grads = jt.grad((out.float32() * upstream.float32()).sum(), qkv)
            tolerance = 3e-3 if dtype == "float16" else 2e-2
            for label, got, expected in zip(("out", "dq", "dk", "dv"), [out] + list(grads), want):
                got.sync()
                self.assertEqual(got.location(), "device", label)
                value = got.float32().numpy()
                self.assertTrue(np.isfinite(value).all(), label)
                np.testing.assert_allclose(value, expected, rtol=tolerance,
                                           atol=tolerance * max(np.abs(expected).max(), 1e-8),
                                           err_msg=label)
            jt.sync_all(True)
            if expect_fused:
                self.assertIsNotNone(route)
                self.assertEqual(route.__module__, self.kernel.__name__)
            else:
                self.assertIsNone(route)
            self.assertEqual(str(out.dtype), dtype)
        return out.float32().numpy(), grads[0].float32().numpy()

    def test_masked_training_with_empty_rows_and_causal_intersections(self):
        for dtype in ("float16", "bfloat16"):
            for causal in (False, True):
                for kind in ("bool", "additive"):
                    with self.subTest(dtype=dtype, causal=causal, mask=kind):
                        mask = np.random.RandomState(42).rand(1, 1, 33, 47) > 0.25
                        mask[..., 5, :] = False
                        mask[..., 0, 0] = False  # causal leaves this row no key
                        mask[..., 0, 1:] = True
                        if kind == "additive":
                            bias = np.random.RandomState(43).normal(0, .2, mask.shape).astype("float32")
                            mask = np.where(mask, bias, -np.inf).astype("float32")
                        out, dq = self._check_mask_training(dtype, mask, causal)
                        self.assertEqual(np.abs(out[..., 5, :]).max(), 0.)
                        self.assertEqual(np.abs(dq[..., 5, :]).max(), 0.)
                        if causal:
                            self.assertEqual(np.abs(out[..., 0, :]).max(), 0.)
                            self.assertEqual(np.abs(dq[..., 0, :]).max(), 0.)

    def test_padding_masks_broadcast_during_training(self):
        for dtype in ("float16", "bfloat16"):
            for shape in ((47,), (1, 1, 1, 47)):
                for kind in ("bool", "additive"):
                    with self.subTest(dtype=dtype, shape=shape, mask=kind):
                        mask = np.arange(47) < 31
                        if kind == "additive":
                            mask = np.where(mask, 0., -1e9).astype("float32")
                        self._check_mask_training(dtype, mask.reshape(shape))

    def test_large_finite_row_offsets_keep_the_explicit_softmax_backward(self):
        for dtype in ("float16", "bfloat16"):
            with self.subTest(dtype=dtype):
                mask = np.zeros((33, 47), "float32")
                mask[5, :] = np.finfo(np.float32).min
                self._check_mask_training(dtype, mask, expect_fused=False)

    def test_autocast_keeps_the_float32_bias_abi(self):
        mask = np.where(np.arange(47) < 31, 0., -1e9).astype("float32")
        for dtype, preference in (("float16", jt.amp_flags.prefer16),
                                  ("bfloat16", jt.amp_flags.prefer_bfloat16)):
            with self.subTest(dtype=dtype), jt.flag_scope(amp_reg=preference):
                self._check_mask_training(dtype, mask.reshape(1, 1, 1, 47))
                bias = self.kernel._MASK_TERMS[0][2]
                self.assertEqual(str(bias.dtype), "float32")
                bias.sync()
                self.assertEqual(bias.location(), "device")

    def test_grouped_heads_with_a_constant_mask(self):
        mask = (np.arange(47) < 31).reshape(1, 1, 1, 47)
        self._check_mask_training("float16", mask, grouped=True)

    def test_a_learned_mask_is_not_served_without_its_gradient(self):
        q = jt.random((1, 2, 33, 64)).float16()
        k = jt.random((1, 2, 47, 64)).float16()
        mask = jt.zeros((33, 47), dtype="float32").start_grad()
        self.assertIsNone(self.kernel._cudnn_fused_attention(q, k, k, attn_mask=mask))

    def test_masked_training_replays_with_changed_inputs(self):
        def step(q, k, v, mask):
            out = scaled_dot_product_attention(q, k, v, attn_mask=mask)
            return (out,) + tuple(jt.grad(out.float32().sum(), [q, k, v]))
        with forbid_backend_fallbacks():
            for dtype in ("float16", "bfloat16"):
                with self.subTest(dtype=dtype):
                    run = jt.capture_step(step)
                    tolerance = 3e-3 if dtype == "float16" else 2e-2
                    for feed in range(4):
                        rng = np.random.RandomState(50 + feed)
                        arrays = [rng.normal(0, .5, shape).astype("float32") for shape in
                                  ((1, 2, 33, 64), (1, 2, 47, 64), (1, 2, 47, 64))]
                        mask = (np.arange(47) < 31 + feed).reshape(1, 1, 1, 47)
                        tensors = [jt.array(t).cast(dtype) for t in arrays] + [jt.array(mask)]
                        values = [t.float32().numpy() for t in tensors[:3]]
                        for tensor in tensors:
                            tensor.sync()
                        want = _reference(*values, mask, False, np.ones_like(values[0]))
                        result = run(*tensors)
                        for label, got, expected in zip(("out", "dq", "dk", "dv"), result, want):
                            got.sync()
                            self.assertEqual(got.location(), "device", label)
                            np.testing.assert_allclose(
                                got.float32().numpy(), expected, rtol=tolerance,
                                atol=tolerance * max(np.abs(expected).max(), 1e-8), err_msg=label)
                    self.assertEqual(run.stats["captured"], 1)
                    self.assertGreater(run.stats["replayed"], 0)
            jt.sync_all(True)
        # Mask admission reads a scalar only on the eager fused path;
        # tracing retains the existing graph-only composite implementation.

    def test_an_updated_mask_does_not_reuse_the_old_bias(self):
        rng = np.random.RandomState(44)
        arrays = [rng.randn(*shape).astype("float32") for shape in
                  ((1, 2, 33, 64), (1, 2, 47, 64), (1, 2, 47, 64))]
        with jt.no_grad(), forbid_backend_fallbacks():
            qkv = [jt.array(t).float16() for t in arrays]
            values = [t.float32().numpy() for t in qkv]
            mask = jt.ones((1, 1, 1, 47), dtype="bool")
            first = scaled_dot_product_attention(*qkv, attn_mask=mask).float32().numpy()
            visible = (np.arange(47) < 23).reshape(1, 1, 1, 47)
            mask.update(jt.array(visible))
            second = scaled_dot_product_attention(*qkv, attn_mask=mask).float32().numpy()
            want = _reference(*values, visible, False, np.zeros_like(values[0]))[0]
            np.testing.assert_allclose(second, want, atol=3e-3, rtol=3e-3)
            self.assertGreater(np.abs(first-second).max(), 0.01)

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
