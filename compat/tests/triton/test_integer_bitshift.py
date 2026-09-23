"""Packed mask arithmetic must retain integer Triton dtypes in the bridge."""

import numpy as np
import jittor as jt


def test_int32_loaded_mask_shifts_against_int32_offsets():
    import triton
    import triton.language as tl

    @triton.jit
    def shift_kernel(values, output, BLOCK: tl.constexpr):
        offsets = tl.arange(0, BLOCK)
        mask = tl.load(values + offsets)
        shifted = (mask[:, None] >> offsets[None, :]) & 1
        tl.store(output + offsets, tl.sum(shifted, axis=1))

    values = jt.array(np.array([1, 2, 4, 8], dtype=np.int32))
    output = jt.zeros((4,), dtype="int32")
    shift_kernel[(1,)](values, output, BLOCK=4)
    np.testing.assert_array_equal(output.numpy(), np.array([1, 1, 1, 1], dtype=np.int32))


def test_int32_masked_load_with_other_zero_keeps_integer_dtype():
    import triton
    import triton.language as tl

    @triton.jit
    def masked_shift(values, output, BLOCK: tl.constexpr):
        offsets = tl.arange(0, BLOCK)
        packed = tl.load(values + offsets, mask=offsets < 2, other=0)
        bits = (packed[:, None] >> tl.arange(0, 32)[None, :]) & 1
        tl.store(output + offsets, tl.sum(bits, axis=1))

    values = jt.array(np.array([1, 2, 4, 8], dtype=np.int32))
    output = jt.zeros((4,), dtype="int32")
    masked_shift[(1,)](values, output, BLOCK=4)
    np.testing.assert_array_equal(output.numpy(), np.array([1, 1, 0, 0], dtype=np.int32))
