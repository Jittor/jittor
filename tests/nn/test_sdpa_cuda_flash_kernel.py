# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""The CUDA flash kernel declines cleanly, so absent flash nothing changes.

`nn.scaled_dot_product_attention` now has a CUDA implementation backed by the
flash-attention bridge. That is a public behaviour change on a hot path, and
the property that makes it safe is not "flash is faster" -- it is that every
case the kernel cannot serve returns None, because `try_dispatch` hands None
straight back and the caller takes the same math path it always did.

So these assert the declines, not the speed. They need neither a GPU nor a
flash checkout, which is the point: the decline path is what runs on the
machines that have neither.
"""
import unittest
from types import SimpleNamespace

import jittor as jt

from jittor.backends.cuda.kernels.nn.flash_attention_cuda import (
    _flash_scaled_dot_product_attention as _flash_sdpa,
    _template_dim,
)


def _qkv(shape, dtype="float16"):
    return tuple(jt.random(shape).cast(dtype) for _ in range(3))


class TestFlashKernelDeclines(unittest.TestCase):
    def test_a_mask_declines(self):
        q, k, v = _qkv((1, 2, 8, 64))
        mask = jt.zeros((8, 8), dtype="bool")
        self.assertIsNone(_flash_sdpa(q, k, v, attn_mask=mask))

    def test_dropout_declines(self):
        q, k, v = _qkv((1, 2, 8, 64))
        self.assertIsNone(_flash_sdpa(q, k, v, dropout_p=0.1))

    def test_a_non_four_dimensional_input_declines(self):
        q, k, v = _qkv((2, 8, 64))
        self.assertIsNone(_flash_sdpa(q, k, v))

    def test_float32_declines(self):
        # Registered only for float16/bfloat16; the kernels are half precision.
        q, k, v = _qkv((1, 2, 8, 64), dtype="float32")
        self.assertIsNone(_flash_sdpa(q, k, v))

    def test_unequal_head_counts_decline(self):
        # Grouped-query attention needs a head broadcast this kernel does not
        # do; declining is cheaper than getting it subtly wrong.
        q = jt.random((1, 8, 16, 64)).cast("float16")
        k = jt.random((1, 2, 16, 64)).cast("float16")
        v = jt.random((1, 2, 16, 64)).cast("float16")
        self.assertIsNone(_flash_sdpa(q, k, v))

    def test_causal_with_unequal_lengths_declines(self):
        # flash puts a causal mask in the bottom-right corner when the lengths
        # differ; the math lowering puts it top-left. Same shape, different
        # answer -- so this case has to decline.
        q = jt.random((1, 2, 8, 64)).cast("float16")
        k = jt.random((1, 2, 16, 64)).cast("float16")
        v = jt.random((1, 2, 16, 64)).cast("float16")
        self.assertIsNone(_flash_sdpa(q, k, v, is_causal=True))

    def test_a_head_dimension_with_no_template_declines(self):
        self.assertIsNone(_template_dim(257))
        q, k, v = _qkv((1, 2, 8, 257))
        self.assertIsNone(_flash_sdpa(q, k, v))

    def test_a_supported_head_dimension_rounds_up_to_its_template(self):
        # The guard above must not be vacuous: ordinary sizes do resolve.
        self.assertEqual(_template_dim(64), 64)
        self.assertEqual(_template_dim(48), 64)
        self.assertEqual(_template_dim(80), 96)


class TestNoBackendMeansNoChange(unittest.TestCase):
    """Without a flash build the kernel declines rather than raising."""

    def _with_bridge(self, bridge):
        import jittor.backends.cuda.kernels.nn.flash_attention_cuda as module
        original = module._bridge
        module._bridge = lambda: bridge
        self.addCleanup(setattr, module, "_bridge", original)

    def test_no_backend_declines(self):
        self._with_bridge(SimpleNamespace(
            enabled=lambda: True,
            required=lambda: False,
            last_error=lambda: "no source root",
            load_backend_for=lambda dim, dtype: (None, None)))
        q, k, v = _qkv((1, 2, 8, 64))
        self.assertIsNone(_flash_sdpa(q, k, v))

    def test_disabled_declines(self):
        self._with_bridge(SimpleNamespace(enabled=lambda: False))
        q, k, v = _qkv((1, 2, 8, 64))
        self.assertIsNone(_flash_sdpa(q, k, v))

    def test_required_turns_an_absent_backend_into_an_error(self):
        # The caller asked to hear about it. A silent slow path is the failure
        # mode `JITTOR_FLASH_ATTN_JITTOR_REQUIRED` exists to prevent.
        self._with_bridge(SimpleNamespace(
            enabled=lambda: True,
            required=lambda: True,
            last_error=lambda: "no source root",
            load_backend_for=lambda dim, dtype: (None, None)))
        q, k, v = _qkv((1, 2, 8, 64))
        with self.assertRaises(RuntimeError) as caught:
            _flash_sdpa(q, k, v)
        self.assertIn("flash", str(caught.exception).lower())


class TestTheKernelIsPublished(unittest.TestCase):
    def test_importing_jittor_nn_registers_the_cuda_kernel(self):
        """Published by import, not from inside the call.

        It lived in `attention.py`'s call path first. That broke
        `tests/structure/nn/test_attention_softmax_dispatch.py`, which loads
        `attention.py` against a NumPy stand-in for jittor where
        `jittor.backends` is a stub with no such module -- and paying an import
        check on every attention call was never the right shape anyway.
        `jittor/nn/backends/__init__.py` publishes the other CUDA nn kernels
        exactly this way.
        """
        import jittor.nn  # noqa: F401  -- the import is the registration
        from jittor._runtime import dispatch
        self.assertIn("nn.scaled_dot_product_attention",
                      dispatch._registered_ops)


if __name__ == "__main__":
    unittest.main()
