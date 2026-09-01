"""vLLM's decoder-layer patches, against both conventions vLLM has used."""

import types
import unittest

import numpy as np

import jittor as jt

from jittor.compat.vllm import layers


def _rotary_module(returns_cache):
    """A stand-in for vLLM's rotary module, in either calling convention.

    ``_match_cos_sin_cache_dtype`` aligns the cache with the query. Older vLLM
    only mutates the attribute and returns nothing; newer versions hand the
    aligned cache back so the caller need not read it again.
    """

    module = types.ModuleType("vllm.model_executor.layers.rotary_embedding.base")

    class RotaryEmbedding:
        def __init__(self, head_size, rotary_dim, max_position, is_neox_style):
            self.head_size = head_size
            self.rotary_dim = rotary_dim
            self.is_neox_style = is_neox_style
            frequency = 1.0 / (10000 ** (
                np.arange(0, rotary_dim, 2, dtype="float32") / rotary_dim))
            angles = np.outer(
                np.arange(max_position, dtype="float32"), frequency)
            self.cos_sin_cache = jt.array(
                np.concatenate([np.cos(angles), np.sin(angles)], axis=-1))

        def _match_cos_sin_cache_dtype(self, query):
            return self.cos_sin_cache if returns_cache else None

        def forward_native(self, positions, query, key=None):
            raise AssertionError("the patch should not have deferred here")

    module.RotaryEmbedding = RotaryEmbedding
    return module


def _rotate(returns_cache):
    module = _rotary_module(returns_cache)
    assert layers.patch_rotary_embedding(module)
    layer = module.RotaryEmbedding(8, 8, 16, True)
    positions = jt.array(np.arange(3).astype("int64"))
    query = jt.array(np.random.RandomState(0).randn(3, 16).astype("float32"))
    key = jt.array(np.random.RandomState(1).randn(3, 8).astype("float32"))
    # vLLM's CustomOp.forward dispatches to one of these two; the patch
    # replaces both so the choice cannot change the answer.
    assert layer.forward_cuda.__func__ is layer.forward_native.__func__
    return layer.forward_cuda(positions, query, key)


class TestRotaryEmbeddingPatch(unittest.TestCase):
    def test_both_calling_conventions_rotate_the_same_way(self):
        old_query, old_key = _rotate(returns_cache=False)
        new_query, new_key = _rotate(returns_cache=True)
        self.assertLess(
            np.abs(old_query.numpy() - new_query.numpy()).max(), 1e-6)
        self.assertLess(np.abs(old_key.numpy() - new_key.numpy()).max(), 1e-6)

    def test_position_zero_is_left_unrotated(self):
        module = _rotary_module(returns_cache=False)
        layers.patch_rotary_embedding(module)
        layer = module.RotaryEmbedding(8, 8, 16, True)
        query = jt.array(np.random.RandomState(2).randn(1, 8).astype("float32"))
        key = jt.array(np.random.RandomState(3).randn(1, 8).astype("float32"))
        rotated_query, rotated_key = layer.forward_cuda(
            jt.array(np.zeros(1, dtype="int64")), query, key)
        self.assertLess(
            np.abs(rotated_query.numpy() - query.numpy()).max(), 1e-6)
        self.assertLess(np.abs(rotated_key.numpy() - key.numpy()).max(), 1e-6)

    def test_patching_twice_changes_nothing(self):
        module = _rotary_module(returns_cache=True)
        self.assertTrue(layers.patch_rotary_embedding(module))
        self.assertFalse(layers.patch_rotary_embedding(module))


class TestQwen3AttentionPatch(unittest.TestCase):
    def test_non_acl_execution_keeps_original_forward(self):
        module = types.ModuleType("vllm.model_executor.models.qwen3")

        class Qwen3Attention:
            def __init__(self):
                self.q_norm = types.SimpleNamespace(weight=jt.ones(8))
                self.k_norm = types.SimpleNamespace(weight=jt.ones(8))

            def forward(self, positions, hidden_states):
                del positions
                return hidden_states + 1.0

        module.Qwen3Attention = Qwen3Attention
        self.assertTrue(layers.patch_qwen3_attention(module))
        self.assertFalse(layers.patch_qwen3_attention(module))
        hidden = jt.zeros((2, 8))
        output = module.Qwen3Attention().forward(jt.zeros((2,)), hidden)
        np.testing.assert_array_equal(output.numpy(), np.ones((2, 8)))


if __name__ == "__main__":
    unittest.main()
