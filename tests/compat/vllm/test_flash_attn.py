"""The attention entry points vLLM imports, against a plain-numpy reference."""

import unittest

import numpy as np

import jittor as jt

torch = jt
from jittor.compat.vllm import flash_attn


def _softmax(values, axis=-1):
    shifted = values - values.max(axis=axis, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=axis, keepdims=True)


def _attend(query, key, value, causal):
    """query [Lq, H, D], key/value [Lk, H, D] -> [Lq, H, D], bottom-right aligned."""
    scale = query.shape[-1] ** -0.5
    scores = np.einsum("qhd,khd->hqk", query, key) * scale
    if causal:
        length_q, length_k = query.shape[0], key.shape[0]
        offset = length_k - length_q
        rows = np.arange(length_q)[:, None]
        columns = np.arange(length_k)[None, :]
        scores = np.where(columns <= rows + offset, scores, -np.inf)
    return np.einsum("hqk,khd->qhd", _softmax(scores), value)


class TestVarlenAttention(unittest.TestCase):
    def test_matches_a_per_sequence_reference(self):
        np.random.seed(3)
        lengths = [3, 5, 1]
        heads, kv_heads, dim = 4, 2, 8
        total = sum(lengths)
        query = np.random.randn(total, heads, dim).astype("float32")
        key = np.random.randn(total, kv_heads, dim).astype("float32")
        value = np.random.randn(total, kv_heads, dim).astype("float32")
        starts = np.concatenate([[0], np.cumsum(lengths)]).astype("int32")

        got = flash_attn.flash_attn_varlen_func(
            jt.array(query), jt.array(key), jt.array(value),
            jt.array(starts), jt.array(starts), causal=True).numpy()

        expected = np.concatenate([
            _attend(query[starts[i]:starts[i + 1]],
                    np.repeat(key[starts[i]:starts[i + 1]], heads // kv_heads, 1),
                    np.repeat(value[starts[i]:starts[i + 1]], heads // kv_heads, 1),
                    causal=True)
            for i in range(len(lengths))])
        np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-4)

    def test_a_single_sequence_needs_no_concatenation(self):
        query = jt.array(np.random.randn(4, 2, 8).astype("float32"))
        starts = jt.array(np.array([0, 4], dtype="int32"))
        got = flash_attn.flash_attn_varlen_func(
            query, query, query, starts, starts, causal=False)
        self.assertEqual(tuple(got.shape), (4, 2, 8))


class TestDecodeAgainstAPagedCache(unittest.TestCase):
    def test_attends_over_exactly_the_cached_prefix(self):
        np.random.seed(5)
        blocks, block_size, kv_heads, dim, heads = 6, 4, 2, 8, 4
        key_cache = np.random.randn(blocks, block_size, kv_heads, dim).astype("float32")
        value_cache = np.random.randn(blocks, block_size, kv_heads, dim).astype("float32")
        # Two sequences whose pages are deliberately not contiguous.
        table = np.array([[0, 3, 0, 0], [4, 1, 2, 0]], dtype="int32")
        lengths = [5, 9]
        query = np.random.randn(2, 1, heads, dim).astype("float32")

        got = flash_attn.flash_attn_with_kvcache(
            jt.array(query), jt.array(key_cache), jt.array(value_cache),
            cache_seqlens=jt.array(np.array(lengths, dtype="int32")),
            block_table=jt.array(table)).numpy()

        expected = []
        for row, length in enumerate(lengths):
            used = -(-length // block_size)
            pages = table[row, :used]
            keys = key_cache[pages].reshape(-1, kv_heads, dim)[:length]
            values = value_cache[pages].reshape(-1, kv_heads, dim)[:length]
            expected.append(_attend(
                query[row].reshape(1, heads, dim),
                np.repeat(keys, heads // kv_heads, 1),
                np.repeat(values, heads // kv_heads, 1), causal=False))
        np.testing.assert_allclose(
            got.reshape(2, 1, heads, dim),
            np.stack(expected).reshape(2, 1, heads, dim), rtol=1e-4, atol=1e-4)


class TestCacheWrites(unittest.TestCase):
    def test_v1_write_lands_in_the_slots_it_is_given(self):
        blocks, block_size, kv_heads, dim = 4, 4, 2, 8
        cache = jt.zeros((blocks, 2, block_size, kv_heads, dim), dtype="float32")
        key = jt.array(np.random.randn(3, kv_heads, dim).astype("float32"))
        value = jt.array(np.random.randn(3, kv_heads, dim).astype("float32"))
        slots = jt.array(np.array([0, 5, 11], dtype="int32"))

        flash_attn.reshape_and_cache_kv_v1(key, value, cache, slots)

        written = cache.numpy()
        for token, slot in enumerate([0, 5, 11]):
            block, offset = slot // block_size, slot % block_size
            np.testing.assert_allclose(written[block, 0, offset], key.numpy()[token])
            np.testing.assert_allclose(written[block, 1, offset], value.numpy()[token])

    def test_separate_caches_take_the_same_slot_mapping(self):
        blocks, block_size, kv_heads, dim = 3, 4, 2, 8
        key_cache = jt.zeros((blocks, block_size, kv_heads, dim), dtype="float32")
        value_cache = jt.zeros((blocks, block_size, kv_heads, dim), dtype="float32")
        key = jt.array(np.random.randn(2, kv_heads, dim).astype("float32"))
        value = jt.array(np.random.randn(2, kv_heads, dim).astype("float32"))

        # A negative slot marks a padding token, which is skipped.
        flash_attn.reshape_and_cache_flash(
            key, value, key_cache, value_cache,
            jt.array(np.array([6, -1], dtype="int32")))

        np.testing.assert_allclose(
            key_cache.numpy()[1, 2], key.numpy()[0], rtol=1e-6, atol=1e-6)
        self.assertEqual(float(jt.abs(value_cache).sum()),
                         float(jt.abs(value[0]).sum()))


class TestTheBundleItPublishes(unittest.TestCase):
    def test_the_flash_attention_bundle_answers_both_import_paths(self):
        import sys

        flash_attn.install()
        for name in ("vllm.vllm_flash_attn",
                     "vllm.vllm_flash_attn.flash_attn_interface"):
            module = sys.modules[name]
            self.assertIs(module.flash_attn_varlen_func,
                          flash_attn.flash_attn_varlen_func)
            self.assertIs(module.flash_attn_with_kvcache,
                          flash_attn.flash_attn_with_kvcache)

    def test_a_submodule_it_does_not_carry_still_imports(self):
        import importlib
        import sys
        import types

        flash_attn.install()
        # In a real run vLLM itself owns the parent package; here it only has
        # to exist for the import machinery to descend past it.
        created = "vllm" not in sys.modules
        if created:
            parent = types.ModuleType("vllm")
            parent.__path__ = []
            sys.modules["vllm"] = parent
        try:
            rotary = importlib.import_module(
                "vllm.vllm_flash_attn.layers.rotary")
            self.assertTrue(callable(rotary.apply_rotary_emb))
        finally:
            if created:
                for name in list(sys.modules):
                    if name == "vllm" or name.startswith("vllm."):
                        del sys.modules[name]


if __name__ == "__main__":
    unittest.main()
