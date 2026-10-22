"""A cast of a Python number is built as the converted constant.

`UnaryOp::forward_before_construct` folds ``unary(number, dtype)`` into an
array of the target dtype when the conversion is exact: the cast of a
one-element array is a kernel of its own (its output may not fuse into the
broadcast that reads it), and an integer ``clamp`` converts both bounds this
way. The oracle is the same cast of a constant that is not a Python number --
``jt.array(v)`` is not marked as one -- which still runs the cast kernel.
Inexact conversions are not folded and must keep answering what the kernel
answers.
"""

from _helpers import capability as _test_capability

import unittest

import numpy as np

import jittor as jt


_TARGETS = ("bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32",
            "uint64", "float32", "float64")
_VALUES = (0, 1, 3, -2, 127, 255, 300, -129, 2**24 + 1, 2**31 - 1, True, False)


class _Folding:
    use_cuda = 0

    def setUp(self):
        self._scope = jt.flag_scope(use_cuda=self.use_cuda, auto_convert_64_to_32=0)
        self._scope.__enter__()

    def tearDown(self):
        self._scope.__exit__(None, None, None)

    def test_a_number_cast_matches_the_cast_kernel(self):
        for target in _TARGETS:
            for value in _VALUES:
                with self.subTest(target=target, value=value):
                    folded = jt.unary(value, target)
                    kernel = jt.array(value).cast(target)
                    self.assertEqual(str(folded.dtype), str(kernel.dtype))
                    self.assertEqual(folded.shape, kernel.shape)
                    np.testing.assert_array_equal(folded.numpy(), kernel.numpy())

    def test_integer_clamp(self):
        for dtype in ("int8", "int16", "int32", "int64", "uint8"):
            a = np.array([0, 1, 3, 7, 100], dtype)
            for low, high in ((0, 6), (2, 50), (None, 6), (2, None)):
                with self.subTest(dtype=dtype, low=low, high=high):
                    got = jt.array(a).clamp(low, high)
                    self.assertEqual(str(got.dtype), dtype)
                    np.testing.assert_array_equal(got.numpy(), np.clip(a, low, high))

    def test_a_scalar_that_records_a_gradient_keeps_its_cast(self):
        w = jt.array(3.0)
        w.start_grad()
        g = jt.grad((w.int32().float32() * w).sum(), w)
        np.testing.assert_array_equal(g.numpy(), np.array(3.0, "float32"))


class TestNumberCastFoldingCPU(_Folding, unittest.TestCase):
    use_cuda = 0


@unittest.skipIf(not _test_capability.check_accelerator('cuda', backend=jt).enabled, "No CUDA found")
class TestNumberCastFoldingCUDA(_Folding, unittest.TestCase):
    use_cuda = 1


if __name__ == "__main__":
    unittest.main()
