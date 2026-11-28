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
import gc
import sys
import unittest
from unittest.mock import patch
import weakref

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

    def test_native_gradient_owner_uses_fresh_inputs_and_cotangents(self):
        rng = np.random.RandomState(247)
        shape = (2, 2, 33, 8)
        before = jt.core.backend_fallback_count()
        with jt.flag_scope(keep_graph=0, backend_fallback="error"), \
                patch.object(self.kernel._FusedAttentionF32, "apply",
                             side_effect=AssertionError("Python gradient owner used")):
            for _ in range(2):
                q, k, v, cotangent = [rng.randn(*shape).astype("float32") for _ in range(4)]
                expected = _reference(q, k, v, False, cotangent)
                inputs = [jt.array(a) for a in (q, k, v)]
                for tensor in inputs:
                    tensor.start_grad()
                output = self.kernel._fused_attention_f32(*inputs)
                gradients = jt.grad((output * jt.array(cotangent)).sum(), inputs)
                for actual, reference in zip([output] + list(gradients), expected):
                    actual.sync()
                    self.assertEqual(actual.location(), "device")
                    self.assertGreaterEqual(actual.device_id, 0)
                    np.testing.assert_allclose(actual.numpy(), reference, rtol=1e-4,
                                               atol=1e-5 * np.abs(reference).max())
        self.assertEqual(jt.core.backend_fallback_count(), before)

    def test_half_training_uses_tiled_opmath_when_half_kernel_declines(self):
        from jittor._runtime.dispatch import override_kernel

        rng = np.random.RandomState(287)
        shape = (1, 2, 31, 16)
        q, k, v, dout = [rng.normal(0, 0.2, shape).astype("float16") for _ in range(4)]
        keep = np.tril(np.ones((31, 31), dtype=bool))
        keep[0] = False
        expected = _reference(q, k, v, False, dout, keep)
        inputs = [jt.array(array).start_grad() for array in (q, k, v)]
        calls = []

        def tiled_if_fp32(query, key, value, **kwargs):
            if str(query.dtype) != "float32":
                return None
            calls.append(str(query.dtype))
            return self.kernel._fused_attention_f32(query, key, value, **kwargs)

        with override_kernel("nn.scaled_dot_product_attention", "cuda", None), \
                override_kernel("nn.fused_attention", "cuda", tiled_if_fp32), \
                jt.flag_scope(backend_fallback="error"):
            result = scaled_dot_product_attention(*inputs, attn_mask=jt.array(keep))
            grads = jt.grad((result * jt.array(dout)).sum(), inputs)
        self.assertEqual(calls, ["float32"])
        for actual, reference in zip([result] + list(grads), expected):
            actual.sync()
            self.assertEqual(actual.location(), "device")
            self.assertEqual(str(actual.dtype), "float16")
            np.testing.assert_allclose(actual.numpy(), reference, rtol=2e-3, atol=4e-4)

    def test_mask_callback_handles_retired_owner_and_stale_reference(self):
        cache = self.kernel._LAST_MASK
        saved = cache[:]
        previous_hook = sys.unraisablehook
        errors = []

        class Owner:
            pass

        owner = Owner()
        ref = weakref.ref(owner, self.kernel._forget_mask)
        try:
            self.kernel._LAST_MASK = None
            sys.unraisablehook = errors.append
            del owner
            gc.collect()
            self.assertEqual(errors, [])
            self.assertIsNone(ref())
            self.kernel._LAST_MASK = cache
            first, second = object(), object()
            cache[:] = [second, None, None, "marker"]
            self.kernel._forget_mask(first)
            self.assertIs(cache[0], second)
            self.kernel._forget_mask(second)
            self.assertEqual(cache, [None, None, None, None])
        finally:
            sys.unraisablehook = previous_hook
            self.kernel._LAST_MASK = cache
            cache[:] = saved

    def test_nearly_saturated_probabilities_backward(self):
        # These original parity samples have tiny Q/K gradients. Reconstructing
        # scores with a different scale/dot order breaks their normalization.
        for causal, seeds in ((False, (821, 831, 841)),
                              (True, (861, 871, 881))):
            with self.subTest(causal=causal):
                shape = (1, 1, 4, 6)
                q, k, v = (np.random.RandomState(seed).uniform(
                    -9, 9, size=shape).astype("float32") for seed in seeds)
                dout = np.random.RandomState(1234).standard_normal(shape).astype("float32")
                expected = _reference(q, k, v, causal, dout)
                jq, jk, jv = (jt.array(a) for a in (q, k, v))
                calls = []
                original = self.kernel._forward

                def counted(*args):
                    calls.append(1)
                    return original(*args)

                self.kernel._forward = counted
                try:
                    out = scaled_dot_product_attention(jq, jk, jv, is_causal=causal)
                    grads = jt.grad((out * jt.array(dout)).sum(), [jq, jk, jv])
                    actual = [a.numpy().copy() for a in [out] + list(grads)]
                finally:
                    self.kernel._forward = original
                self.assertEqual(calls, [1])
                for index, (got, want) in enumerate(zip(actual, expected)):
                    self.assertTrue(np.isfinite(got).all())
                    error = np.abs(got.astype("float64") - want)
                    # The existing device-parity budgets, without relaxation.
                    net_tol, pe_tol = (2e-4, 2e-3) if index == 0 else (2e-3, 5e-3)
                    self.assertLess(float(error.max() / (np.abs(want).max() + 1e-30)), net_tol)
                    self.assertLess(float((error / (np.abs(want) + 1e-3)).max()), pe_tol)

    def test_mixed_saturated_and_diffuse_rows_backward(self):
        # A block can see both decisions and the last block has unused warps.
        # Check the peaked rows alongside their diffuse neighbors.
        shape = (1, 1, 11, 6)
        rng = np.random.RandomState(507)
        q, k, v = (rng.uniform(-1, 1, size=shape).astype("float32")
                   for _ in range(3))
        q = q.copy()
        q[0, 0, (0, 2, 8)] *= 80
        dout = rng.standard_normal(shape).astype("float32")
        for causal in (False, True):
            with self.subTest(causal=causal):
                expected = _reference(q, k, v, causal, dout)
                jq, jk, jv = (jt.array(a) for a in (q, k, v))
                out = scaled_dot_product_attention(jq, jk, jv, is_causal=causal)
                grads = jt.grad((out * jt.array(dout)).sum(), [jq, jk, jv])
                jt.sync([out, *grads], device_sync=True)
                self.assertTrue(all(t.location() == "device" for t in [out, *grads]))
                actual = [a.numpy().copy() for a in [out] + list(grads)]
                for index, (got, want) in enumerate(zip(actual, expected)):
                    self.assertTrue(np.isfinite(got).all())
                    error = np.abs(got.astype("float64") - want)
                    net_tol, pe_tol = (2e-4, 2e-3) if index == 0 else (2e-3, 5e-3)
                    self.assertLess(float(error.max() / (np.abs(want).max() + 1e-30)), net_tol)
                    self.assertLess(float((error / (np.abs(want) + 1e-3)).max()), pe_tol)

    def test_constant_values_have_zero_query_and_key_gradients(self):
        rng = np.random.RandomState(913)
        shape = (1, 1, 7, 6)
        q, k = (rng.uniform(-9, 9, size=shape).astype("float32") for _ in range(2))
        v = np.broadcast_to(rng.randn(1, 1, 1, 6), shape).astype("float32")
        dout = rng.randn(*shape).astype("float32")
        for causal in (False, True):
            with self.subTest(causal=causal):
                jq, jk, jv = (jt.array(a) for a in (q, k, v))
                out = scaled_dot_product_attention(jq, jk, jv, is_causal=causal)
                dq, dk = jt.grad((out * jt.array(dout)).sum(), [jq, jk])
                # With identical values the result is independent of Q and K.
                np.testing.assert_allclose(dq.numpy(), 0., atol=1e-10, rtol=0.)
                np.testing.assert_allclose(dk.numpy(), 0., atol=1e-10, rtol=0.)

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

    def test_pending_setitem_inputs_keep_their_backward_storage(self):
        # A concatenation leaves a pending setitem. Materialization can reuse
        # its source storage, changing the provisional token after the custom
        # attention tape has saved the input.
        from jittor.autograd import EXPLICIT_REQUIRES_GRAD, policy_scope

        arrays = [np.arange(64, dtype=np.float32).reshape(1, 4, 4, 4) / scale
                  for scale in (100, 110, 120, 130)]
        value = np.arange(128, dtype=np.float32).reshape(1, 4, 4, 8) / 140
        query = np.concatenate(arrays[:2], axis=-1)
        key = np.concatenate(arrays[2:], axis=-1)
        expected = _reference(query, key, value, True, np.ones_like(value))

        with policy_scope(EXPLICIT_REQUIRES_GRAD):
            parts = [jt.array(array).requires_grad_(True) for array in arrays]
            value_var = jt.array(value).requires_grad_(True)
            query_var = jt.concat(parts[:2], dim=-1)
            key_var = jt.concat(parts[2:], dim=-1)
            out = self.kernel._fused_attention_f32(
                query_var, key_var, value_var, is_causal=True)
            loss = out.sum()
            float(loss)  # Run the forward before constructing its backward.
            grads = jt.grad(loss, parts + [value_var])

        for actual, want in zip(grads, (
                expected[1][..., :4], expected[1][..., 4:],
                expected[2][..., :4], expected[2][..., 4:], expected[3])):
            np.testing.assert_allclose(actual.numpy(), want, rtol=1e-4, atol=1e-5)

    # Masks. Transformers passes an explicit one whenever
    # `torch.compiler.is_compiling()`, a padded batch always does, and both
    # used to decline to the path that writes every score out.
    def test_a_key_padding_mask(self):
        mask = np.ones((2, 1, 1, 47), bool)
        mask[0, ..., 30:] = False
        mask[1, ..., :5] = False
        self._check(2, 3, 33, 47, 40, False, mask)

    def test_an_additive_key_padding_mask(self):
        # What Transformers expands an attention mask into: one row per batch,
        # the dtype's minimum where padded. 47 keys: rows off the 16-byte
        # alignment the vector loads need, and a partial last group of four.
        mask = (np.random.RandomState(3).randn(2, 1, 1, 47) * 0.5).astype("float32")
        mask[0, ..., 30:] = np.finfo(np.float32).min
        self._check(2, 3, 33, 47, 40, False, mask)

    def test_a_dense_additive_mask(self):
        # The same mask as Transformers passes it under compilation, one row
        # per query, on lengths that keep every row aligned for vector loads.
        mask = (np.random.RandomState(4).randn(2, 1, 64, 64) * 0.5).astype("float32")
        mask[0, ..., 40:] = np.finfo(np.float32).min
        mask[1, ..., :9] = np.finfo(np.float32).min
        # The forward remains uniform here; backward must retain 1 / key_count
        # even when adding log(key_count) to finfo.min would round it away.
        mask[1, :, 7, :] = np.finfo(np.float32).min
        self._check(2, 3, 64, 64, 64, False, mask)

    def test_a_sliding_window_additive_mask(self):
        # Sliding-window additive masks (as ModernBERT builds with finfo.min or
        # -inf) leave most 64x64 key tiles completely masked out. Include an
        # all-finfo.min query row spanning skipped tiles to verify fallback.
        idx = np.arange(192)
        window = np.abs(idx[:, None] - idx[None, :]) <= 32
        for sentinel in (np.finfo(np.float32).min, -np.inf):
            with self.subTest(sentinel=sentinel):
                mask = np.where(window, 0.0, sentinel).astype("float32")
                mask = np.broadcast_to(mask, (2, 1, 192, 192)).copy()
                if np.isfinite(sentinel):
                    mask[1, :, 5, :] = sentinel
                self._check(2, 2, 192, 192, 64, False, mask)

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

    def test_seq_major_inputs_are_read_in_place(self):
        # q/k/v as `x.view(b, s, h, d).transpose(1, 2)`: the kernels read the
        # [b, s, h, d] tensors, write the output and every gradient back in
        # them, and nothing is transposed.
        b, h, s, d = 2, 3, 40, 32
        rng = np.random.RandomState(9)
        q, k, v, dout = (rng.randn(b, h, s, d).astype("float32") for _ in range(4))
        want = _reference(q, k, v, True, dout)
        sources = [jt.array(np.ascontiguousarray(t.transpose(0, 2, 1, 3))) for t in (q, k, v)]
        out = scaled_dot_product_attention(
            *(t._storage_permute((0, 2, 1, 3)) for t in sources), is_causal=True)
        grads = jt.grad((out * jt.array(dout)).sum(), sources)
        np.testing.assert_allclose(out.numpy(), want[0], rtol=1e-4, atol=1e-5)
        for name, got, expected in zip("qkv", grads, want[1:]):
            np.testing.assert_allclose(got.numpy(), expected.transpose(0, 2, 1, 3), rtol=1e-4,
                                       atol=1e-5 * np.abs(expected).max(), err_msg="d" + name)

    def test_materialized_transpose_keeps_its_gradient_target(self):
        rng = np.random.RandomState(311)
        q = rng.randn(2, 2, 5, 8).astype("float32")
        k, v = (rng.randn(2, 2, 7, 8).astype("float32") for _ in range(2))
        cotangent = rng.randn(*q.shape).astype("float32")
        expected = _reference(q, k, v, False, cotangent)
        sources = [jt.array(array.transpose(0, 2, 1, 3).copy()) for array in (q, k, v)]
        for source in sources:
            source.start_grad()
        with jt.flag_scope(transpose_storage_view=0, missing_grad_error=1):
            inputs = [jt.ops.transpose(source, (0, 2, 1, 3)) for source in sources]
            self.assertTrue(all(value._storage_is_contiguous() for value in inputs))
            output = scaled_dot_product_attention(*inputs)
            gradients = jt.grad((output * jt.array(cotangent)).sum(), inputs)
            for actual, reference in zip([output, *gradients], expected):
                np.testing.assert_allclose(actual.numpy(), reference, rtol=1e-4,
                                           atol=1e-5 * np.abs(reference).max())

    def _check_grouped(self, b, hq, hk, s, d, causal, mask=None, seq_major=False):
        """Grouped-query attention against the closed form on repeated heads."""
        group = hq // hk
        rng = np.random.RandomState(hq * 10 + hk)
        q, dout = (rng.randn(b, hq, s, d).astype("float32") for _ in range(2))
        k, v = (rng.randn(b, hk, s, d).astype("float32") for _ in range(2))
        out_ref, dq_ref, dk_rep, dv_rep = _reference(
            q, np.repeat(k, group, 1), np.repeat(v, group, 1), causal, dout, mask)
        dk_ref = dk_rep.reshape(b, hk, group, s, d).sum(2)
        dv_ref = dv_rep.reshape(b, hk, group, s, d).sum(2)
        seen = []
        original = self.kernel._forward

        def counted(query, key, value, call):
            seen.append((tuple(key.shape), call.group))
            return original(query, key, value, call)
        self.kernel._forward = counted
        self.addCleanup(setattr, self.kernel, "_forward", original)
        if seq_major:
            sources = [jt.array(np.ascontiguousarray(t.transpose(0, 2, 1, 3))) for t in (q, k, v)]
            jq, jk, jv = (t.transpose(1, 2) for t in sources)
        else:
            sources = [jt.array(t) for t in (q, k, v)]
            jq, jk, jv = sources
        out = scaled_dot_product_attention(
            jq, jk, jv, attn_mask=None if mask is None else jt.array(mask).stop_grad(),
            is_causal=causal, enable_gqa=True)
        grads = jt.grad((out * jt.array(dout)).sum(), sources)
        # The key and value heads were read in place, never repeated.
        self.assertEqual([g for _, g in seen], [group])
        self.assertEqual(seen[0][0][2 if seq_major else 1], hk)
        want = (dq_ref, dk_ref, dv_ref)
        np.testing.assert_allclose(out.numpy(), out_ref, rtol=1e-4, atol=1e-5)
        for name, got, expected in zip("qkv", grads, want):
            if seq_major:
                expected = expected.transpose(0, 2, 1, 3)
            np.testing.assert_allclose(got.numpy(), expected, rtol=1e-4,
                                       atol=1e-5 * np.abs(expected).max(), err_msg="d" + name)

    def test_grouped_query_heads_are_read_in_place(self):
        self._check_grouped(2, 4, 2, 37, 32, False)
        self._check_grouped(1, 8, 2, 70, 64, True)

    def test_grouped_query_heads_with_a_mask(self):
        rng = np.random.RandomState(3)
        mask = rng.rand(2, 1, 33, 33) > 0.3
        mask[..., 0] = True
        self._check_grouped(2, 6, 3, 33, 40, False, mask=mask)

    def test_grouped_query_heads_seq_major(self):
        self._check_grouped(2, 4, 1, 40, 32, True, seq_major=True)

    def test_a_pending_repeat_of_the_heads_is_read_from_its_source(self):
        # Transformers' `repeat_kv`, as it runs whenever it builds a mask: the
        # repeated copy is never made, and the gradients reach the source.
        b, hq, hk, s, d = 2, 6, 2, 29, 32
        group = hq // hk
        rng = np.random.RandomState(21)
        q, dout = (rng.randn(b, hq, s, d).astype("float32") for _ in range(2))
        k, v = (rng.randn(b, hk, s, d).astype("float32") for _ in range(2))
        out_ref, dq_ref, dk_rep, dv_rep = _reference(
            q, np.repeat(k, group, 1), np.repeat(v, group, 1), True, dout)
        seen = []
        original = self.kernel._forward

        def counted(query, key, value, call):
            seen.append((tuple(key.shape), call.group))
            return original(query, key, value, call)
        self.kernel._forward = counted
        self.addCleanup(setattr, self.kernel, "_forward", original)

        def repeat(t):
            return t[:, :, None, :, :].broadcast((b, hk, group, s, d)).reshape((b, hq, s, d))
        jq, jk, jv = (jt.array(t) for t in (q, k, v))
        out = scaled_dot_product_attention(jq, repeat(jk * 1), repeat(jv * 1), is_causal=True)
        grads = jt.grad((out * jt.array(dout)).sum(), [jq, jk, jv])
        self.assertEqual(seen, [((b, hk, s, d), group)])
        np.testing.assert_allclose(out.numpy(), out_ref, rtol=1e-4, atol=1e-5)
        want = (dq_ref, dk_rep.reshape(b, hk, group, s, d).sum(2),
                dv_rep.reshape(b, hk, group, s, d).sum(2))
        for name, got, expected in zip("qkv", grads, want):
            np.testing.assert_allclose(got.numpy(), expected, rtol=1e-4,
                                       atol=1e-5 * np.abs(expected).max(), err_msg="d" + name)

    def _fixed_seed(self):
        original = self.kernel._Call.__init__

        def init(call, *args):
            original(call, *args)
            if call.seed is not None:
                call.seed = jt.array(np.array([12345, 678], "int32")).stop_grad()
        self.kernel._Call.__init__ = init
        self.addCleanup(setattr, self.kernel._Call, "__init__", original)

    def test_dropout_gradients_match_finite_differences(self):
        # With the seed fixed the dropped forward is a function; its
        # gradients must be that function's.
        self._fixed_seed()
        rng = np.random.RandomState(5)
        q, k, v, dout = (rng.randn(1, 2, 20, 16).astype("float64") for _ in range(4))

        def loss(q_, k_, v_):
            out = scaled_dot_product_attention(
                *(jt.array(t.astype("float32")) for t in (q_, k_, v_)), dropout_p=0.3)
            return float((out.numpy().astype("float64") * dout).sum())
        jq, jk, jv = (jt.array(t.astype("float32")) for t in (q, k, v))
        out = scaled_dot_product_attention(jq, jk, jv, dropout_p=0.3)
        grads = [g.numpy() for g in jt.grad((out * jt.array(dout.astype("float32"))).sum(),
                                            [jq, jk, jv])]
        eps = 1e-2
        for which, grad in enumerate(grads):
            for index in ((0, 0, 3, 5), (0, 1, 17, 0), (0, 1, 9, 11)):
                args = [q, k, v]
                plus, minus = [a.copy() for a in args], [a.copy() for a in args]
                plus[which][index] += eps
                minus[which][index] -= eps
                numeric = (loss(*plus) - loss(*minus)) / (2 * eps)
                self.assertAlmostEqual(grad[index], numeric, delta=2e-2 * max(1, abs(numeric)),
                                       msg="grad %s at %s" % ("qkv"[which], index))

    def test_dropout_keeps_the_expectation(self):
        rng = np.random.RandomState(6)
        q, k, v = (jt.array(rng.randn(1, 2, 64, 32).astype("float32")) for _ in range(3))
        plain = scaled_dot_product_attention(q, k, v).numpy()
        runs = [scaled_dot_product_attention(q, k, v, dropout_p=0.2).numpy() for _ in range(256)]
        self.assertFalse(np.allclose(runs[0], runs[1]))
        mean = np.mean(runs, 0)
        self.assertLess(np.abs(mean - plain).mean(), 0.05 * np.abs(plain).mean() + 1e-3)

    def test_what_it_declines(self):
        q = jt.random((1, 2, 8, 256))
        self.assertIsNone(self.kernel._fused_attention_f32(q, q, q))
        q = jt.random((1, 2, 8, 64))
        self.assertIsNone(self.kernel._fused_attention_f32(q, q, q, dropout_p=1.0))
        # A learned bias needs its own gradient; a half-precision one is not
        # what this float32 kernel reads; a mask must broadcast.
        self.assertIsNone(self.kernel._fused_attention_f32(
            q, q, q, attn_mask=jt.zeros((8, 8))))
        self.assertIsNone(self.kernel._fused_attention_f32(
            q, q, q, attn_mask=jt.zeros((8, 8)).float16().stop_grad()))
        self.assertIsNone(self.kernel._fused_attention_f32(
            q, q, q, attn_mask=jt.ones((3, 8), dtype="bool")))

    def test_short_inference_fast_path(self):
        rng = np.random.RandomState(42)
        cases = [
            ("chronos_time_add_mask", (4, 12, 35, 64), (4, 12, 35, 64), False, "add_b11k"),
            ("chronos_group_add_mask", (35, 12, 4, 64), (35, 12, 4, 64), False, "add_b1qk"),
            ("d128_causal", (2, 4, 19, 128), (2, 4, 19, 128), True, None),
            ("odd_d_bool_mask", (2, 3, 13, 37), (2, 3, 29, 37), False, "bool_all_false_row"),
            ("long_q_tiny_k", (2, 4, 128, 64), (2, 4, 7, 64), False, None),
            ("sentinel_all_masked", (1, 2, 4, 64), (1, 2, 4, 64), False, "sentinel"),
            ("gqa_short", (2, 8, 17, 64), (2, 2, 23, 64), True, None),
        ]
        with jt.no_grad():
            for name, qshape, kshape, causal, mask_kind in cases:
                with self.subTest(case=name):
                    q = rng.randn(*qshape).astype("float32")
                    k = rng.randn(*kshape).astype("float32")
                    v = rng.randn(*kshape).astype("float32")
                    b, h, lq, d = qshape
                    hk, lk = kshape[1], kshape[2]
                    mask_np = None
                    if mask_kind == "add_b11k":
                        mask_np = np.zeros((b, 1, 1, lk), dtype="float32")
                        mask_np[:, :, :, -5:] = -np.inf
                        mask_np[0, 0, 0, :3] = -3.4028235e+38
                    elif mask_kind == "add_b1qk":
                        mask_np = np.zeros((b, 1, lq, lk), dtype="float32")
                        mask_np[:, :, 0, 2:] = -np.inf
                        mask_np[1, 0, 3, :] = -np.inf
                    elif mask_kind == "bool_all_false_row":
                        mask_np = rng.rand(b, 1, lq, lk) > 0.3
                        mask_np[0, 0, 0, :] = False
                    elif mask_kind == "sentinel":
                        mask_np = np.full((b, 1, lq, lk), -3.4028235e+38, dtype="float32")
                        mask_np[0, 0, 1, 2] = 0.0
                    k_ref = np.repeat(k, h // hk, axis=1) if hk != h else k
                    v_ref = np.repeat(v, h // hk, axis=1) if hk != h else v
                    expected = _reference(q, k_ref, v_ref, causal, np.zeros_like(q), mask_np)[0]

                    jq, jk, jv = (jt.array(a).stop_grad() for a in (q, k, v))
                    jm = jt.array(mask_np).stop_grad() if mask_np is not None else None
                    # Contiguous [B,H,S,D]
                    out = self.kernel._fused_attention_f32(
                        jq, jk, jv, attn_mask=jm, is_causal=causal, enable_gqa=(hk != h))
                    self.assertIsNotNone(out)
                    np.testing.assert_allclose(out.numpy(), expected, rtol=1e-4, atol=1e-5)

                    # Seq-major [B,S,H,D].transpose(0, 2, 1, 3) view
                    jq_s = jt.array(np.ascontiguousarray(q.transpose(0, 2, 1, 3))).stop_grad().transpose(0, 2, 1, 3)
                    jk_s = jt.array(np.ascontiguousarray(k.transpose(0, 2, 1, 3))).stop_grad().transpose(0, 2, 1, 3)
                    jv_s = jt.array(np.ascontiguousarray(v.transpose(0, 2, 1, 3))).stop_grad().transpose(0, 2, 1, 3)
                    out_s = self.kernel._fused_attention_f32(
                        jq_s, jk_s, jv_s, attn_mask=jm, is_causal=causal, enable_gqa=(hk != h))
                    self.assertIsNotNone(out_s)
                    np.testing.assert_allclose(out_s.numpy(), expected, rtol=1e-4, atol=1e-5)

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
