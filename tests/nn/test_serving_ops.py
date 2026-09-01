"""Public serving primitives: fused CUDA path == portable fallback == reference."""

import unittest
from unittest import mock

import numpy as np

import jittor as jt
import jittor.nn.serving_ops as serving_ops


def _rope_cache(max_pos, rotary_dim, dtype="float32"):
    inv = 1.0 / (10000 ** (np.arange(0, rotary_dim, 2) / rotary_dim))
    angles = np.arange(max_pos)[:, None] * inv[None, :]
    return np.concatenate([np.cos(angles), np.sin(angles)], -1).astype(dtype)


class TestSiluAndMul(unittest.TestCase):
    def test_acl_miss_keeps_cuda_backend_reachable(self):
        marker = object()
        missing = object()
        previous = getattr(jt.nn, "_silu_and_mul_acl", missing)
        jt.nn._silu_and_mul_acl = lambda value: None
        try:
            with mock.patch.object(
                serving_ops, "_silu_and_mul_cuda", return_value=marker
            ) as cuda_backend:
                self.assertIs(serving_ops.silu_and_mul(object()), marker)
                cuda_backend.assert_called_once()
        finally:
            if previous is missing:
                del jt.nn._silu_and_mul_acl
            else:
                jt.nn._silu_and_mul_acl = previous

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


if __name__ == "__main__":
    unittest.main()
