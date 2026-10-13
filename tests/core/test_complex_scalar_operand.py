# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""A complex scalar meeting a real Var.

``1j * x`` has to work even though nothing about ``x`` says a complex is
coming. This used to be served by a python wrapper installed over the eight
arithmetic operators, so every ``+``, ``-``, ``*`` and ``/`` in every model
paid a python frame for it. The conversion belongs in the argument
converter, and these tests pin down what it has to produce.

A bare python/numpy complex *scalar* narrows to complex64 here, exactly as a
python float narrows to float32 -- this is the argument converter's literal-
construction default (see ``py_converter.h``'s ``ArrayArgs`` scalar path), not
a width limit of jittor's complex support. jittor also has a native complex128
dtype (``KI-COMPLEX-001`` is closed; see ``docs/notes/complex-dtype.md``):
``dtype_infer``/``binary_dtype_infer`` promote to complex128 when either
*Var* operand actually needs double precision (a complex128 Var, or a
float64/int64-width real Var). That promotion rule is exercised by
``tests/core/test_complex128_native.py``, not here -- this file is only about
the scalar-literal narrowing default.
"""

import unittest

import numpy as np

import jittor as jt


class TestComplexScalarOperand(unittest.TestCase):

    def setUp(self):
        self.x = jt.array(np.array([1.0, 2.0], dtype="float32"))

    def _check(self, got, want):
        self.assertEqual(str(got.dtype), "complex64")
        np.testing.assert_allclose(got.numpy(), want, rtol=1e-6, atol=1e-6)

    def test_python_complex_on_either_side_of_each_operator(self):
        x = self.x
        self._check(1j * x, [1j, 2j])
        self._check(x * 1j, [1j, 2j])
        self._check(x + 1j, [1 + 1j, 2 + 1j])
        self._check(1j + x, [1 + 1j, 2 + 1j])
        self._check(x - 1j, [1 - 1j, 2 - 1j])
        self._check(1j - x, [-1 + 1j, -2 + 1j])
        self._check(x / 1j, [-1j, -2j])
        self._check(1j / x, [1j, 0.5j])

    def test_a_complex_with_both_parts(self):
        self._check((2 + 3j) * self.x, [2 + 3j, 4 + 6j])

    def test_numpy_complex_scalars_narrow_to_complex64(self):
        # complex64 is not a subclass of python complex and reaches the
        # converter as a numpy scalar; complex128 *is* a subclass, and is the
        # one numpy complex type the dtype table cannot name.
        self._check(np.complex64(1j) * self.x, [1j, 2j])
        self._check(np.complex128(1j) * self.x, [1j, 2j])
        self._check(self.x * np.complex128(1j), [1j, 2j])

    def test_a_complex_scalar_is_a_var_on_its_own(self):
        v = jt.array(1 + 2j)
        self.assertEqual(str(v.dtype), "complex64")
        np.testing.assert_allclose(v.numpy(), 1 + 2j, rtol=1e-6)

    def test_real_scalars_are_untouched(self):
        # A real scalar does not lift the tensor's category, so both stay
        # float32 here; the point is only that the complex branch did not
        # capture them.
        for value in (2.0, 2):
            got = self.x * value
            self.assertEqual(str(got.dtype), "float32")
            np.testing.assert_allclose(got.numpy(), [value, 2 * value])


if __name__ == "__main__":
    unittest.main()
