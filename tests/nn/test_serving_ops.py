"""Public serving primitives: fused CUDA path == portable fallback == reference."""

import unittest
from unittest import mock
from types import SimpleNamespace

import numpy as np

import jittor as jt
import jittor.nn.serving_ops as serving_ops
from jittor._runtime.dispatch import dispatch_context, override_kernel


def _rope_cache(max_pos, rotary_dim, dtype="float32"):
    inv = 1.0 / (10000 ** (np.arange(0, rotary_dim, 2) / rotary_dim))
    angles = np.arange(max_pos)[:, None] * inv[None, :]
    return np.concatenate([np.cos(angles), np.sin(angles)], -1).astype(dtype)


class TestSiluAndMul(unittest.TestCase):
    def test_acl_miss_keeps_cuda_backend_reachable(self):
        marker = object()
        with override_kernel(
            "nn.silu_and_mul", dispatch_context().backend, lambda value: None,
        ):
            with mock.patch.object(
                serving_ops, "_silu_and_mul_cuda", return_value=marker
            ) as cuda_backend:
                self.assertIs(serving_ops.silu_and_mul(object()), marker)
                cuda_backend.assert_called_once()

    def test_matches_the_gate_times_value_reference(self):
        raw = np.random.randn(9, 24).astype("float32")
        x = jt.array(raw)
        gate, value = raw[:, :12], raw[:, 12:]
        expected = gate / (1.0 + np.exp(-gate)) * value
        np.testing.assert_allclose(
            jt.nn.silu_and_mul(x).numpy(), expected, rtol=1e-5, atol=1e-5)

    def test_keeps_every_leading_axis(self):
        x = jt.array(np.random.randn(2, 5, 8).astype("float32"))
        self.assertEqual(tuple(jt.nn.silu_and_mul(x).shape), (2, 5, 4))


class TestRmsNorm(unittest.TestCase):
    def test_replaced_placement_query_is_observed(self):
        x, weight = jt.ones((2, 4)), jt.ones(4)
        backend = dispatch_context(x, weight).backend
        result = jt.ones((2, 4))
        context_query = mock.Mock(return_value=SimpleNamespace(backend=backend))
        with mock.patch.object(serving_ops, "dispatch_context", context_query), \
                override_kernel("nn.rms_norm.inference", backend,
                                lambda *args: result):
            self.assertIs(jt.nn.rms_norm(x, weight), result)
        context_query.assert_called_once_with(x, weight)

    @unittest.skipIf(not jt.compiler.has_cuda, "CUDA is required")
    def test_cuda_result_uses_device_storage(self):
        raw = np.array([[0.5, 1., -2., 3.]], dtype="float32")
        with jt.flag_scope(use_cuda=1), jt.no_grad():
            result = jt.nn.rms_norm(jt.array(raw), jt.ones(4), 1e-6)
            result.sync()
            self.assertEqual(result.location(), "device")
            expected = raw / np.sqrt((raw*raw).mean(-1, keepdims=True) + 1e-6)
            np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5, atol=1e-6)

    def test_live_registered_provider_is_called_once(self):
        x = jt.array([[1., 2., 3., 4.]])
        weight = jt.ones(4)
        backend = dispatch_context(x, weight).backend
        calls = []
        for scale in (2., 3.):
            result = x * scale

            def provider(value, gamma, epsilon):
                self.assertIs(value, x)
                self.assertIs(gamma, weight)
                self.assertEqual(epsilon, 1e-6)
                calls.append(scale)
                return result

            with override_kernel("nn.rms_norm.inference", backend, provider):
                self.assertIs(jt.nn.rms_norm(x, weight, 1e-6), result)
                np.testing.assert_array_equal(result.numpy(), [[scale, 2*scale, 3*scale, 4*scale]])
        self.assertEqual(calls, [2., 3.])
        np.testing.assert_array_equal(x.numpy(), [[1., 2., 3., 4.]])

    def test_ineligible_registered_provider_keeps_portable_fallback(self):
        raw = np.array([[1., 2., 3., 4.]], dtype="float32")
        x, weight = jt.array(raw), jt.ones(4)
        backend = dispatch_context(x, weight).backend

        def unavailable(*args):
            raise AssertionError("an ineligible provider must not execute")

        with override_kernel("nn.rms_norm.inference", backend, unavailable,
                             supports=lambda *args: False):
            result = jt.nn.rms_norm(x, weight, 1e-6)
            expected = raw / np.sqrt((raw*raw).mean(-1, keepdims=True) + 1e-6)
            np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5, atol=1e-6)

    def test_registered_provider_errors_are_not_hidden(self):
        x, weight = jt.ones((2, 4)), jt.ones(4)
        backend = dispatch_context(x, weight).backend

        def failing(*args):
            raise RuntimeError("RMS provider failure")

        with override_kernel("nn.rms_norm.inference", backend, failing):
            with self.assertRaisesRegex(RuntimeError, "RMS provider failure"):
                jt.nn.rms_norm(x, weight, 1e-6)

    def test_dual_rms_norm_matches_independent_calls(self):
        rng = np.random.RandomState(20260902)
        first = jt.array(rng.randn(2, 3, 8).astype("float32"))
        second = jt.array(rng.randn(2, 2, 8).astype("float32"))
        first_weight = jt.array((rng.rand(8) + 0.5).astype("float32"))
        second_weight = jt.array((rng.rand(8) + 0.5).astype("float32"))
        actual_first, actual_second = jt.nn.dual_rms_norm(
            first, second, first_weight, second_weight, 1e-6)
        expected_first = jt.nn.rms_norm(first, first_weight, 1e-6)
        expected_second = jt.nn.rms_norm(second, second_weight, 1e-6)
        np.testing.assert_allclose(
            actual_first.numpy(), expected_first.numpy(), rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(
            actual_second.numpy(), expected_second.numpy(), rtol=1e-6, atol=1e-6)

    def test_normalises_over_the_last_axis_and_scales(self):
        raw = np.random.randn(6, 16).astype("float32")
        weight = (np.random.rand(16) + 0.5).astype("float32")
        expected = raw / np.sqrt((raw ** 2).mean(-1, keepdims=True) + 1e-6) * weight
        got = jt.nn.rms_norm(jt.array(raw), jt.array(weight), 1e-6).numpy()
        np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-4)

    def test_fused_add_returns_the_sum_as_the_new_residual(self):
        hidden = np.random.randn(5, 12).astype("float32")
        residual = np.random.randn(5, 12).astype("float32")
        weight = (np.random.rand(12) + 0.5).astype("float32")
        total = hidden + residual
        expected = total / np.sqrt((total ** 2).mean(-1, keepdims=True) + 1e-6) * weight
        normed, carried = jt.nn.fused_add_rms_norm(
            jt.array(hidden), jt.array(residual), jt.array(weight), 1e-6)
        np.testing.assert_allclose(normed.numpy(), expected, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(carried.numpy(), total, rtol=1e-5, atol=1e-5)


class TestRotaryEmbedding(unittest.TestCase):
    def _reference(self, packed, cache, head_size, rotary_dim, tokens, is_neox):
        heads = packed.shape[-1] // head_size
        view = packed.reshape(tokens, heads, head_size).copy()
        half = rotary_dim // 2
        cos = cache[:tokens, :half][:, None, :]
        sin = cache[:tokens, half:rotary_dim][:, None, :]
        span = view[..., :rotary_dim]
        if is_neox:
            first, second = span[..., :half], span[..., half:]
            rotated = np.concatenate(
                [first * cos - second * sin, second * cos + first * sin], -1)
        else:
            first, second = span[..., 0::2], span[..., 1::2]
            rotated = np.stack(
                [first * cos - second * sin, second * cos + first * sin], -1)
            rotated = rotated.reshape(tokens, heads, rotary_dim)
        view[..., :rotary_dim] = rotated
        return view.reshape(packed.shape)

    def test_neox_style_rotates_the_two_halves(self):
        tokens, heads, head_size = 7, 4, 16
        cache = _rope_cache(32, head_size)
        query = np.random.randn(tokens, heads * head_size).astype("float32")
        key = np.random.randn(tokens, 2 * head_size).astype("float32")
        positions = jt.array(np.arange(tokens).astype("int32"))
        got_q, got_k = jt.nn.rotary_embedding(
            positions, jt.array(query), jt.array(key), jt.array(cache),
            head_size=head_size, is_neox=True)
        np.testing.assert_allclose(
            got_q.numpy(),
            self._reference(query, cache, head_size, head_size, tokens, True),
            rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(
            got_k.numpy(),
            self._reference(key, cache, head_size, head_size, tokens, True),
            rtol=1e-4, atol=1e-4)

    def test_gptj_style_rotates_interleaved_pairs(self):
        tokens, heads, head_size = 5, 2, 8
        cache = _rope_cache(16, head_size)
        query = np.random.randn(tokens, heads * head_size).astype("float32")
        positions = jt.array(np.arange(tokens).astype("int32"))
        got, none = jt.nn.rotary_embedding(
            positions, jt.array(query), None, jt.array(cache),
            head_size=head_size, is_neox=False)
        self.assertIsNone(none)
        np.testing.assert_allclose(
            got.numpy(),
            self._reference(query, cache, head_size, head_size, tokens, False),
            rtol=1e-4, atol=1e-4)

    def test_partial_rotary_leaves_the_tail_untouched(self):
        tokens, heads, head_size, rotary_dim = 4, 3, 16, 8
        cache = _rope_cache(16, rotary_dim)
        query = np.random.randn(tokens, heads * head_size).astype("float32")
        positions = jt.array(np.arange(tokens).astype("int32"))
        got, _ = jt.nn.rotary_embedding(
            positions, jt.array(query), None, jt.array(cache),
            head_size=head_size, is_neox=True, rotary_dim=rotary_dim)
        np.testing.assert_allclose(
            got.numpy(),
            self._reference(query, cache, head_size, rotary_dim, tokens, True),
            rtol=1e-4, atol=1e-4)


class TestQkRmsNormRotary(unittest.TestCase):
    """The fused query/key norm-and-rotate pass a serving stack asks for.

    It has no portable fallback -- unpacking heads is the caller's layout -- so
    what the public entry point owes the caller is an honest answer about
    whether a backend has one, and delegation with the arguments unchanged.
    """

    def _with_hook(self, hook):
        scope = override_kernel(
            "nn.grouped_qk_rms_norm_rotary", dispatch_context().backend, hook)
        scope.__enter__()
        self.addCleanup(scope.__exit__, None, None, None)

    def test_absent_backend_is_reported_and_returns_none(self):
        self._with_hook(None)
        self.assertFalse(jt.nn.has_qk_rms_norm_rotary())
        self.assertIsNone(jt.nn.qk_rms_norm_rotary(*([None] * 10)))

    def test_present_backend_receives_every_argument_in_order(self):
        seen = []
        self._with_hook(lambda *args: seen.append(args) or ("q", "k"))
        self.assertTrue(jt.nn.has_qk_rms_norm_rotary())
        arguments = tuple(range(10))
        self.assertEqual(jt.nn.qk_rms_norm_rotary(*arguments), ("q", "k"))
        self.assertEqual(seen, [arguments])

    def test_a_backend_that_declines_these_inputs_returns_none(self):
        self._with_hook(lambda *args: None)
        self.assertTrue(jt.nn.has_qk_rms_norm_rotary())
        self.assertIsNone(jt.nn.qk_rms_norm_rotary(*([None] * 10)))


if __name__ == "__main__":
    unittest.main()
