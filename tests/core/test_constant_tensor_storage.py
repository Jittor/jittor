# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""A constant-filled tensor must own a buffer of its own shape.

``jt.zeros(shape)`` is built as ``array(0) -> broadcast_to(shape)``. An expand
is normally described as strides over the storage it expands, which is what
keeps a real broadcast from materializing -- but applied to a *one-element*
source it makes every element of the result alias the same four bytes. That is
invisible to Jittor's own operators, which read the strides, and wrong for
anything that takes the tensor's pointer and treats it as its shape: a cuSPARSE
SpMM handed ``jt.zeros`` as its dense output wrote a 3x3 matrix and the caller
read back zeros, with no error anywhere (see
``tests/backends/cuda/test_cusparse_op.py``).

The stride of the returned buffer is the narrowest statement of that invariant
and needs no accelerator, so it is asserted here rather than only through the
library tests that happen to depend on it.
"""
import unittest

import numpy as np

import jittor as jt


class TestConstantTensorStorage(unittest.TestCase):

    def _assert_owns_its_buffer(self, var, shape, dtype):
        data = np.asarray(var.data)
        itemsize = np.dtype(dtype).itemsize
        expected = tuple(
            int(np.prod(shape[i + 1:], dtype=np.int64)) * itemsize
            for i in range(len(shape))
        )
        self.assertEqual(data.shape, tuple(shape))
        # The message names the shape rather than the var: formatting a Var
        # goes through a path that rejects a multi-element tensor, and the
        # f-string is built whether or not the assertion fires.
        self.assertEqual(
            data.strides, expected,
            f"{dtype}{tuple(shape)} is backed by strides {data.strides} rather "
            f"than a buffer of its own shape; a zero stride means every element "
            f"aliases one cell")
        self.assertTrue(data.flags["C_CONTIGUOUS"])

    def test_zeros_owns_a_buffer_of_its_shape(self):
        for shape in [(6, 4), (5,), (2, 3, 4), (1, 7)]:
            with self.subTest(shape=shape):
                self._assert_owns_its_buffer(
                    jt.zeros(shape, dtype="float32"), shape, "float32")

    def test_ones_and_full_own_a_buffer_of_their_shape(self):
        for shape in [(6, 4), (3, 3, 3)]:
            with self.subTest(shape=shape):
                self._assert_owns_its_buffer(
                    jt.ones(shape, dtype="float32"), shape, "float32")
                self._assert_owns_its_buffer(
                    jt.full(shape, 2.5, dtype="float32"), shape, "float32")

    def test_constant_dtypes_keep_their_own_buffers(self):
        for dtype in ("float32", "float64", "int32", "int64", "bool"):
            with self.subTest(dtype=dtype):
                self._assert_owns_its_buffer(
                    jt.zeros((4, 3), dtype=dtype), (4, 3), dtype)

    def test_a_multi_element_expand_is_still_a_view(self):
        """The descriptor form is the point for real broadcasts; keep it.

        A row expanded across rows has nothing to materialize, and reporting a
        zero stride for the expanded axis is the correct description of it.
        """
        row = jt.array(np.arange(4, dtype="float32"))
        data = np.asarray(row.broadcast([3, 4]).data)
        self.assertEqual(data.shape, (3, 4))
        self.assertEqual(data.strides[0], 0)
        np.testing.assert_array_equal(data, np.broadcast_to(np.arange(4, dtype="float32"), (3, 4)))

    def test_a_constant_tensor_reads_back_what_was_written_into_it(self):
        """The consequence the library tests hit, without needing a library."""
        z = jt.zeros((3, 3), dtype="float32")
        z[1, 2] = 8.0
        expected = np.zeros((3, 3), dtype="float32")
        expected[1, 2] = 8.0
        np.testing.assert_array_equal(z.numpy(), expected)


if __name__ == "__main__":
    unittest.main()
