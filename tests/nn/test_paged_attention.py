# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# Maintainers:
#     Dun Liang <randonlang@gmail.com>.
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Attention over a paged KV cache, against a written-out reference.

The interesting property is that one function serves three shapes -- prefill,
decode and chunked prefill -- and takes a different route for each: a fused CUDA
decode kernel, a batched path when every request has the same lengths, and a
per-request loop when they differ. All three have to agree with the same
reference, so each case here is written to land on a specific one.
"""
import unittest

import numpy as np

import jittor as jt


def _reference(query, keys, values, scale):
    """Bottom-right causal attention for one request, in float64."""
    span_q, span_k = query.shape[0], keys.shape[0]
    repeats = query.shape[1] // keys.shape[1]
    keys = np.repeat(keys, repeats, axis=1).astype("float64")
    values = np.repeat(values, repeats, axis=1).astype("float64")
    scores = np.einsum("qhd,khd->hqk", query.astype("float64"), keys) * scale
    offset = span_k - span_q
    rows = np.arange(span_q)[:, None]
    cols = np.arange(span_k)[None, :]
    scores = np.where((cols > rows + offset)[None], -np.inf, scores)
    scores = scores - scores.max(-1, keepdims=True)
    weights = np.exp(scores)
    weights /= weights.sum(-1, keepdims=True)
    return np.einsum("hqk,khd->qhd", weights, values)


def _build(request_lengths, num_heads=28, num_kv_heads=4, head_dim=128,
           block_size=16, seed=3):
    """Pack one request per entry of ``request_lengths`` into a paged cache."""
    rng = np.random.RandomState(seed)
    blocks_per = [-(-n // block_size) for n in request_lengths]
    cache = np.zeros(
        (sum(blocks_per) + 1, 2, block_size, num_kv_heads, head_dim), "float32")
    table = np.zeros((len(request_lengths), max(blocks_per)), "int32")
    queries, expected, next_block = [], [], 0
    for i, length in enumerate(request_lengths):
        for j in range(blocks_per[i]):
            table[i, j] = next_block
            next_block += 1
        query = rng.randn(length, num_heads, head_dim).astype("float32") * 0.1
        keys = rng.randn(length, num_kv_heads, head_dim).astype("float32") * 0.1
        values = rng.randn(length, num_kv_heads, head_dim).astype("float32") * 0.1
        padded = blocks_per[i] * block_size
        flat_k = np.zeros((padded, num_kv_heads, head_dim), "float32")
        flat_v = np.zeros((padded, num_kv_heads, head_dim), "float32")
        flat_k[:length], flat_v[:length] = keys, values
        for j in range(blocks_per[i]):
            cache[table[i, j], 0] = flat_k[j * block_size:(j + 1) * block_size]
            cache[table[i, j], 1] = flat_v[j * block_size:(j + 1) * block_size]
        queries.append(query)
        expected.append(_reference(query, keys, values, head_dim ** -0.5))
    starts = np.cumsum([0] + list(request_lengths)).astype("int32")
    return (np.concatenate(queries, 0), cache, starts,
            np.asarray(request_lengths, "int32"), table,
            np.concatenate(expected, 0), head_dim ** -0.5)


class TestPagedAttention(unittest.TestCase):
    def _check(self, request_lengths, tolerance=1e-4):
        query, cache, starts, lengths, table, expected, scale = _build(
            request_lengths)
        with jt.no_grad():
            got = jt.nn.paged_attention(
                jt.array(query), jt.array(cache), jt.array(starts),
                jt.array(lengths), jt.array(table), scale=scale, causal=True)
        got = np.asarray(got.float32().numpy())
        self.assertFalse(np.isnan(got).any(), "paged attention produced NaN")
        np.testing.assert_allclose(got, expected, atol=tolerance, rtol=0)

    def test_single_request_prefill(self):
        self._check([5])

    def test_uniform_batch_takes_the_batched_path(self):
        # Same lengths throughout: one gather and two batched matmuls.
        self._check([7, 7, 7, 7])

    def test_ragged_batch_falls_back_to_the_loop(self):
        # Differing lengths cannot share a mask, so this must take the loop and
        # still agree with the reference.
        self._check([3, 11, 1, 20])

    def test_spans_several_blocks(self):
        self._check([40, 40])

    def test_reshape_and_cache_round_trips(self):
        block_size, heads, dim = 16, 4, 8
        cache = jt.zeros((4, 2, block_size, heads, dim), "float32")
        rng = np.random.RandomState(11)
        key = rng.randn(3, heads, dim).astype("float32")
        value = rng.randn(3, heads, dim).astype("float32")
        # slots 0, 1 and 17 -- the last one lands in the second block.
        slots = np.asarray([0, 1, 17], "int32")
        with jt.no_grad():
            jt.nn.reshape_and_cache(
                jt.array(key), jt.array(value), cache, jt.array(slots))
        stored = np.asarray(cache.numpy())
        np.testing.assert_allclose(stored[0, 0, 0], key[0], atol=0, rtol=0)
        np.testing.assert_allclose(stored[0, 1, 1], value[1], atol=0, rtol=0)
        np.testing.assert_allclose(stored[1, 0, 1], key[2], atol=0, rtol=0)

    def test_reshape_and_cache_skips_negative_slots(self):
        cache = jt.zeros((2, 2, 4, 1, 2), "float32")
        key = jt.array([[[1.0, 2.0]], [[9.0, 9.0]], [[3.0, 4.0]]])
        value = key + 10.0
        slots = jt.array([0, -1, 5]).int32()
        with jt.no_grad():
            jt.nn.reshape_and_cache(key, value, cache, slots, slots=[0, -1, 5])
        stored = np.asarray(cache.numpy())
        np.testing.assert_array_equal(stored[0, 0, 0], key.numpy()[0])
        np.testing.assert_array_equal(stored[1, 1, 1], value.numpy()[2])
        self.assertEqual(np.count_nonzero(stored), 8)


if __name__ == "__main__":
    unittest.main()
