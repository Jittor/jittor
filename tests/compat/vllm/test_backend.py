"""The paged KV cache layout the attention patch declares to vLLM."""

import types
import unittest

from jittor.compat.vllm import backend


def _module(shape_fn, stride_fn=None):
    module = types.ModuleType("vllm.v1.attention.backends.flash_attn")

    class FlashAttentionBackend:
        get_kv_cache_shape = staticmethod(shape_fn)
        if stride_fn is not None:
            get_kv_cache_stride_order = staticmethod(stride_fn)

    module.FlashAttentionBackend = FlashAttentionBackend
    return module


def _kv_first(num_blocks, block_size, num_kv_heads, head_size,
              cache_dtype_str="auto"):
    return (2, num_blocks, block_size, num_kv_heads, head_size)


def _block_major(num_blocks, block_size, num_kv_heads, head_size,
                 cache_dtype_str="auto"):
    return (num_blocks, 2, block_size, num_kv_heads, head_size)


class TestCacheLayoutDeclaration(unittest.TestCase):
    def test_a_kv_first_backend_is_retold_block_major(self):
        module = _module(_kv_first, lambda: (0, 1, 3, 2, 4))
        self.assertTrue(backend.declare_cache_layout(module))
        self.assertEqual(
            module.FlashAttentionBackend.get_kv_cache_shape(11, 16, 5, 7),
            (11, 2, 16, 5, 7))
        # The permutation described the layout that was just replaced.
        self.assertEqual(
            module.FlashAttentionBackend.get_kv_cache_stride_order(),
            (0, 1, 2, 3, 4))

    def test_a_block_major_backend_is_left_alone(self):
        module = _module(_block_major, lambda: (0, 1, 3, 2, 4))
        self.assertFalse(backend.declare_cache_layout(module))
        self.assertEqual(
            module.FlashAttentionBackend.get_kv_cache_shape(11, 16, 5, 7),
            (11, 2, 16, 5, 7))
        self.assertEqual(
            module.FlashAttentionBackend.get_kv_cache_stride_order(),
            (0, 1, 3, 2, 4))

    def test_the_declaration_carries_the_cache_dtype_keyword(self):
        module = _module(_kv_first)
        backend.declare_cache_layout(module)
        self.assertEqual(
            module.FlashAttentionBackend.get_kv_cache_shape(
                3, 16, 5, 7, cache_dtype_str="fp8"),
            (3, 2, 16, 5, 7))

    def test_applying_it_twice_changes_nothing(self):
        module = _module(_kv_first)
        self.assertTrue(backend.declare_cache_layout(module))
        self.assertFalse(backend.declare_cache_layout(module))
        self.assertEqual(
            module.FlashAttentionBackend.get_kv_cache_shape(3, 16, 5, 7),
            (3, 2, 16, 5, 7))

    def test_a_module_without_the_backend_is_ignored(self):
        self.assertFalse(backend.declare_cache_layout(types.ModuleType("empty")))

    def test_a_backend_that_cannot_answer_is_ignored(self):
        def refuses(*args, **kwargs):
            raise ValueError("Block size must be a multiple of 16.")

        self.assertFalse(backend.declare_cache_layout(_module(refuses)))


if __name__ == "__main__":
    unittest.main()
