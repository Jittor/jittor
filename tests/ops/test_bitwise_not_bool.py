# ***************************************************************
# Copyright (c) 2026 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""`bitwise_not` on bool is logical negation, not an integer complement.

The generated kernel spells this operator `~x`. On a C++ ``bool`` that promotes
to ``int`` first, so ``~true`` is ``-2`` and ``~false`` is ``-1`` -- both
non-zero, and the cast back to ``bool`` turns either into ``true``. The operator
therefore returned ``True`` for every element regardless of its input, with no
error: a silent wrong result.

Nothing caught it because nothing ever passed a bool. The OpInfo entry declares
``dtypes=_INT`` and its sample builder coerces through ``_int_dtype``, and the
shared unary battery checks `bitwise_not` only on an ``int32`` array. This file
owns the dtype those two skip.
"""

import unittest

import numpy as np

import jittor as jt
from _helpers import capability as _test_capability


BOOL_INPUT = np.array([True, False, True, False, False, True])
INTEGER_INPUTS = (
    np.array([-1, 2, 3, 0], dtype="int32"),
    np.array([0, 1, 2, 255], dtype="uint8"),
    np.array([-1, 2, 3, 0], dtype="int8"),
)


class TestBitwiseNotBool(unittest.TestCase):
    """NumPy is the reference; the expectation is never another Jittor call."""

    def _check(self, array):
        # Read the dtype back from the Var rather than assuming the array's:
        # `auto_convert_64_to_32` narrows 64-bit input at construction time
        # (KI-DTYPE-002), which is a factory policy and not this operator's
        # business. Comparing against numpy on the dtype Jittor actually holds
        # keeps this test about `bitwise_not`.
        var = jt.array(array)
        held = np.asarray(array).astype(str(var.dtype))
        got = jt.bitwise_not(var).numpy()
        expected = np.bitwise_not(held)
        self.assertEqual(str(got.dtype), str(var.dtype),
                         "bitwise_not must preserve its input dtype")
        np.testing.assert_array_equal(
            got, expected,
            err_msg="bitwise_not(%s) disagrees with numpy" % array.dtype)

    def test_bool_is_logical_negation(self):
        self._check(BOOL_INPUT)

    def test_bool_result_depends_on_its_input(self):
        """The defect returned a constant, so pin that separately.

        An implementation that ignores its input can still satisfy an
        elementwise comparison against a constant-valued fixture; requiring the
        two halves to differ is what makes the constant answer impossible.
        """
        allocated = jt.bitwise_not(jt.array(np.array([True, False]))).numpy()
        self.assertFalse(bool(allocated[0]), "~True must be False")
        self.assertTrue(bool(allocated[1]), "~False must be True")

    def test_integer_widths_are_unchanged(self):
        for array in INTEGER_INPUTS:
            with self.subTest(dtype=str(array.dtype)):
                self._check(array)

    def test_int64_width_is_preserved_when_narrowing_is_off(self):
        """64-bit needs the narrowing policy off, or the width never reaches the op."""
        source = np.array([-1, 2, 3, 0], dtype="int64")
        with jt.flag_scope(auto_convert_64_to_32=0):
            var = jt.array(source)
            self.assertEqual(str(var.dtype), "int64")
            got = jt.bitwise_not(var).numpy()
        self.assertEqual(str(got.dtype), "int64")
        np.testing.assert_array_equal(got, np.bitwise_not(source))

    @unittest.skipIf(not _test_capability.check_accelerator('cuda', backend=jt).enabled,
                     "No CUDA found")
    @jt.flag_scope(use_cuda=1)
    def test_bool_and_integers_on_cuda(self):
        """CUDA builds its own operator table, so it needs its own assertion."""
        self._check(BOOL_INPUT)
        self._check(INTEGER_INPUTS[0])


if __name__ == "__main__":
    unittest.main()
