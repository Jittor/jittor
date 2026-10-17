"""The attention entry points vLLM imports, against a plain-numpy reference."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import jittor as jt

torch = jt
from jittor_adapters.vllm import flash_attn


class TestHostMetadataLifetime(unittest.TestCase):
    def test_acl_default_skips_fence_when_auto_replay_disabled(self):
        class QueryStarts(list):
            reads = 0

            def numpy(self):
                self.reads += 1
                return np.asarray(self)

        query_starts = QueryStarts([0, 1])
        metadata = SimpleNamespace(slot_mapping=[128],
                                   query_start_loc=query_starts, seq_lens=[11])
        fake_jt = SimpleNamespace(
            compiler=SimpleNamespace(has_acl=True),
            flags=SimpleNamespace(use_acl=1, auto_graph_replay=0))
        with patch.object(flash_attn, 'jt', fake_jt), \
                patch.dict('os.environ', {
                    'JITTOR_VLLM_NPU_DIRECT_METADATA_READBACK': '1',
                }, clear=True):
            cache = {}
            flash_attn._host_metadata(metadata, cache)
            flash_attn._host_metadata(metadata, cache)
        self.assertEqual(query_starts.reads, 0)

    def test_acl_default_fences_when_auto_replay_enabled(self):
        class QueryStarts(list):
            reads = 0

            def numpy(self):
                self.reads += 1
                return np.asarray(self)

        query_starts = QueryStarts([0, 1])
        metadata = SimpleNamespace(slot_mapping=[128],
                                   query_start_loc=query_starts, seq_lens=[11])
        fake_jt = SimpleNamespace(
            compiler=SimpleNamespace(has_acl=True),
            flags=SimpleNamespace(use_acl=1, auto_graph_replay=1))
        with patch.object(flash_attn, 'jt', fake_jt), \
                patch.dict('os.environ', {
                    'JITTOR_VLLM_NPU_DIRECT_METADATA_READBACK': '1',
                }, clear=True):
            cache = {}
            flash_attn._host_metadata(metadata, cache)
            flash_attn._host_metadata(metadata, cache)
        self.assertEqual(query_starts.reads, 1)

    def test_acl_direct_readback_skips_extra_cpu_var(self):
        class QueryStarts(list):
            reads = 0

            def numpy(self):
                self.reads += 1
                return np.asarray(self)

        query_starts = QueryStarts([0, 1])
        metadata = SimpleNamespace(slot_mapping=[128],
                                   query_start_loc=query_starts, seq_lens=[11])
        fake_jt = SimpleNamespace(
            compiler=SimpleNamespace(has_acl=True),
            flags=SimpleNamespace(use_acl=1))
        host_reads = []
        with patch.object(flash_attn, 'jt', fake_jt), \
                patch.object(flash_attn, 'host_lengths',
                             side_effect=lambda value: host_reads.append(id(value)) or list(value)), \
                patch.dict('os.environ', {
                    'JITTOR_VLLM_NPU_SAFE_METADATA_READBACK': '1',
                    'JITTOR_VLLM_NPU_DIRECT_METADATA_READBACK': '1',
                }):
            cache = {}
            first = flash_attn._host_metadata(metadata, cache)
            second = flash_attn._host_metadata(metadata, cache)
        self.assertIs(first, second)
        self.assertEqual(query_starts.reads, 1)
        self.assertEqual(host_reads.count(id(query_starts)), 1)

    def test_acl_rechecks_query_starts_after_first_layer_only(self):
        metadata = SimpleNamespace(slot_mapping=[128],
                                   query_start_loc=[0, 1], seq_lens=[11])
        reads = []

        def read(value):
            reads.append(id(value))
            return list(value)

        fake_jt = SimpleNamespace(
            compiler=SimpleNamespace(has_acl=True),
            flags=SimpleNamespace(use_acl=1))
        with patch.object(flash_attn, 'jt', fake_jt), \
                patch.object(flash_attn, 'host_lengths', side_effect=read), \
                  patch.dict('os.environ', {
                      'JITTOR_VLLM_NPU_SAFE_METADATA_READBACK': '1',
                      'JITTOR_VLLM_NPU_DIRECT_METADATA_READBACK': '0',
                  }):
            cache = {}
            first = flash_attn._host_metadata(metadata, cache)
            second = flash_attn._host_metadata(metadata, cache)
        self.assertIs(first, second)
        self.assertEqual(reads.count(id(metadata.query_start_loc)), 2)
        self.assertEqual(reads.count(id(metadata.slot_mapping)), 1)
        self.assertEqual(reads.count(id(metadata.seq_lens)), 1)

    def test_cuda_keeps_one_read_per_forward(self):
        metadata = SimpleNamespace(slot_mapping=[128],
                                   query_start_loc=[0, 1], seq_lens=[11])
        reads = []
        fake_jt = SimpleNamespace(
            compiler=SimpleNamespace(has_acl=False),
            flags=SimpleNamespace(use_acl=0))
        with patch.object(flash_attn, 'jt', fake_jt), \
                patch.object(flash_attn, 'host_lengths',
                             side_effect=lambda value: reads.append(id(value)) or list(value)):
            cache = {}
            flash_attn._host_metadata(metadata, cache)
            flash_attn._host_metadata(metadata, cache)
        self.assertEqual(len(reads), 3)


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
    def test_f32_prefill_auto_policy_handles_torch_shim_dtype(self):
        policy = flash_attn._f32_prefill_policy
        self.assertTrue(policy(128, "torch.float16", 128, "auto"))
        self.assertTrue(policy(32, "float16", 64, "auto"))
        self.assertTrue(policy(512, "torch.float16", 128, "auto"))
        self.assertFalse(policy(513, "torch.float16", 128, "auto"))
        self.assertFalse(policy(128, "torch.bfloat16", 128, "auto"))
        self.assertFalse(policy(128, "torch.float16", 129, "auto"))
        self.assertFalse(policy(128, "torch.float16", 128, "0"))
        self.assertFalse(policy(128, "torch.float16", 128, "unknown"))
        self.assertTrue(policy(128, "torch.bfloat16", 128, "1"))

    def test_direct_prefill_only_for_plain_decoder_semantics(self):
        impl = SimpleNamespace(
            attn_type=SimpleNamespace(name="DECODER"),
            sliding_window=(-1, -1),
            alibi_slopes=None,
            logits_soft_cap=0,
            sinks=None,
            kv_sharing_target_layer_name=None,
            kv_cache_dtype="auto",
            dcp_world_size=1,
        )
        metadata = SimpleNamespace(
            causal=True, use_cascade=False, common_prefix_len=0,
            mm_prefix_range_tensor=None,
        )
        self.assertTrue(flash_attn._plain_decoder_prefill(impl, metadata))
        for name, value in (
            ("sliding_window", (127, 0)),
            ("alibi_slopes", object()),
            ("logits_soft_cap", 1.0),
            ("sinks", object()),
            ("kv_sharing_target_layer_name", "other"),
            ("kv_cache_dtype", "fp8"),
            ("dcp_world_size", 2),
        ):
            old = getattr(impl, name)
            setattr(impl, name, value)
            self.assertFalse(flash_attn._plain_decoder_prefill(impl, metadata))
            setattr(impl, name, old)
        impl.attn_type = SimpleNamespace(name="ENCODER")
        self.assertFalse(flash_attn._plain_decoder_prefill(impl, metadata))
        impl.attn_type.name = "DECODER"
        for name, value in (
            ("causal", False),
            ("use_cascade", True),
            ("common_prefix_len", 10),
            ("mm_prefix_range_tensor", object()),
        ):
            old = getattr(metadata, name)
            setattr(metadata, name, value)
            self.assertFalse(flash_attn._plain_decoder_prefill(impl, metadata))
            setattr(metadata, name, old)

    def test_direct_full_prefill_gqa_and_chunk_guard(self):
        rng = np.random.default_rng(17)
        tokens, heads, kv_heads, dim = 12, 4, 2, 8
        query = rng.normal(size=(tokens, heads, dim)).astype("float16")
        key = rng.normal(size=(tokens, kv_heads, dim)).astype("float16")
        value = rng.normal(size=(tokens, kv_heads, dim)).astype("float16")
        tensors = tuple(jt.array(t) for t in (query, key, value))
        scale = dim ** -0.5

        got = flash_attn._direct_full_prefill(
            *tensors, [0, tokens], [tokens], True, scale)
        expected = _attend(
            query.astype("float32"),
            np.repeat(key.astype("float32"), heads // kv_heads, axis=1),
            np.repeat(value.astype("float32"), heads // kv_heads, axis=1),
            causal=True)
        np.testing.assert_allclose(got.numpy(), expected, atol=2e-2, rtol=2e-2)
        self.assertIsNone(flash_attn._direct_full_prefill(
            *tensors, [0, tokens], [tokens + 1], True, scale))
        self.assertIsNone(flash_attn._direct_full_prefill(
            *tensors, [0, 6, tokens], [6, 6], True, scale))
        self.assertIsNone(flash_attn._direct_full_prefill(
            tensors[0][:1], tensors[1][:1], tensors[2][:1],
            [0, 1], [1], True, scale))

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


class TestGraphDecodeGuard(unittest.TestCase):
    def test_only_full_single_request_one_token_shape_is_accepted(self):
        class StaticTensor:
            def __init__(self, shape):
                self.shape = shape

            def numel(self):
                result = 1
                for size in self.shape:
                    result *= size
                return result

        metadata = SimpleNamespace(
            max_query_len=1,
            block_table=StaticTensor((1, 4)),
            query_start_loc=StaticTensor((2,)),
            seq_lens=StaticTensor((1,)),
        )
        self.assertTrue(flash_attn._graph_single_request_decode(metadata, 1))
        self.assertFalse(flash_attn._graph_single_request_decode(metadata, 2))
        metadata.max_query_len = 2
        self.assertFalse(flash_attn._graph_single_request_decode(metadata, 1))
        metadata.max_query_len = 1
        metadata.block_table = StaticTensor((4, 4))
        self.assertFalse(flash_attn._graph_single_request_decode(metadata, 1))
        metadata.block_table = StaticTensor((1, 4))
        metadata.query_start_loc = StaticTensor((3,))
        self.assertFalse(flash_attn._graph_single_request_decode(metadata, 1))


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
        # Everything the install owns, taken after it ran: this test may only
        # remove what it adds. Deleting every `vllm.*` name took the bundle
        # `install()` published with it, and its runtime hook then had no
        # module left to roll back -- reported by the next file's fixture as
        # "runtime hook lost module", one file away from the cause.
        published = {name for name in sys.modules
                     if name == "vllm" or name.startswith("vllm.")}
        # In a real run vLLM itself owns the parent package; here it only has
        # to exist for the import machinery to descend past it.
        if "vllm" not in sys.modules:
            parent = types.ModuleType("vllm")
            parent.__path__ = []
            sys.modules["vllm"] = parent
        try:
            rotary = importlib.import_module(
                "vllm.vllm_flash_attn.layers.rotary")
            self.assertTrue(callable(rotary.apply_rotary_emb))
        finally:
            for name in list(sys.modules):
                if (name == "vllm" or name.startswith("vllm.")) \
                        and name not in published:
                    del sys.modules[name]


if __name__ == "__main__":
    unittest.main()
