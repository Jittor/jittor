"""The paged KV cache layout the attention patch declares to vLLM."""

import types
import unittest

from jittor.compat.vllm import backend


def _module(shape_fn, stride_fn=None, head_sizes=None):
    module = types.ModuleType("vllm.v1.attention.backends.flash_attn")

    class FlashAttentionBackend:
        get_kv_cache_shape = staticmethod(shape_fn)
        if stride_fn is not None:
            get_kv_cache_stride_order = staticmethod(stride_fn)
        if head_sizes is not None:
            @classmethod
            def get_supported_head_sizes(cls):
                return list(head_sizes)

            @classmethod
            def validate_head_size(cls, head_size):
                if head_size not in cls.get_supported_head_sizes():
                    raise ValueError("Head size %s is not supported by "
                                     "FlashAttention." % head_size)

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


class TestHeadSizeDeclaration(unittest.TestCase):
    def test_a_head_size_the_kernel_refuses_is_accepted(self):
        module = _module(_block_major, head_sizes=[32, 64, 128])
        backend_class = module.FlashAttentionBackend
        with self.assertRaises(ValueError):
            backend_class.validate_head_size(16)
        self.assertTrue(backend.declare_head_sizes(module))
        backend_class.validate_head_size(16)   # no longer refused
        self.assertIn(16, backend_class.get_supported_head_sizes())

    def test_the_ceiling_is_still_enforced(self):
        module = _module(_block_major, head_sizes=[32, 64, 128])
        backend.declare_head_sizes(module)
        module.FlashAttentionBackend.validate_head_size(256)
        with self.assertRaises(ValueError):
            module.FlashAttentionBackend.validate_head_size(257)
        with self.assertRaises(ValueError):
            module.FlashAttentionBackend.validate_head_size(0)

    def test_a_backend_that_validates_elsewhere_is_left_alone(self):
        module = _module(_block_major)
        self.assertFalse(backend.declare_head_sizes(module))

    def test_declare_backend_covers_both(self):
        module = _module(_kv_first, head_sizes=[32])
        self.assertTrue(backend.declare_backend(module))
        self.assertEqual(
            module.FlashAttentionBackend.get_kv_cache_shape(3, 16, 5, 7),
            (3, 2, 16, 5, 7))
        module.FlashAttentionBackend.validate_head_size(16)


if __name__ == "__main__":
    unittest.main()
