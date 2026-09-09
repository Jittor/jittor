# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""``%`` and ``//`` must agree on which way division rounds.

C's ``%`` takes the sign of the dividend; Python, NumPy and Torch take the sign
of the divisor. Jittor's ``//`` already floored -- ``_floor_divide`` exists for
exactly that -- while ``%`` fell through to the bare C operator for every
integer width. The two halves of one operation therefore used opposite
conventions and ``(a // b) * b + a % b`` returned ``-9`` for ``a = -7``.

Floats were never affected: their branch was written as ``x - floor(x/y)*y``
from the start. So the defect was dtype-dependent -- ``(-7.0) % 2.0`` gave
``1.0`` and ``(-7) % 2`` gave ``-1`` -- which is why no float-only test could
see it, and why the OpInfo entries that inherit ``dtypes=floating_types()``
could not either.

The identity is the assertion that matters. Comparing against NumPy alone would
pass for any convention NumPy happens to share; the identity is what makes the
two operators answerable to each other.
"""

from _helpers import capability as _test_capability

import unittest

import numpy as np

import jittor as jt


SIGNED = ("int8", "int16", "int32", "int64")
UNSIGNED = ("uint8",)

#: Both signs on both sides, and a case where the remainder is zero.
DIVIDENDS = (-7, -5, -4, -1, 0, 1, 4, 5, 7)
DIVISORS = (2, 3, -2, -3)


def _has_cuda():
    return bool(_test_capability.check_accelerator('cuda', backend=jt).enabled)


class _ModContract:

    device_flag = 0

    def _pair(self, dividends, divisors, dtype):
        a = np.array(dividends, dtype=dtype)
        b = np.array(divisors, dtype=dtype)
        return a, b, jt.array(a), jt.array(b)

    def test_signed_mod_matches_numpy(self):
        with jt.flag_scope(use_cuda=self.device_flag):
            for dtype in SIGNED:
                for divisor in DIVISORS:
                    a, b, A, B = self._pair(
                        DIVIDENDS, [divisor] * len(DIVIDENDS), dtype)
                    got = jt.mod(A, B).numpy()
                    np.testing.assert_array_equal(
                        got, a % b,
                        err_msg="%s %% %d diverged from numpy" % (dtype, divisor))

    def test_the_division_identity_holds(self):
        # The point of the contract: `//` and `%` have to round the same way.
        # A test that only compared `%` against numpy would still pass if `//`
        # drifted the other direction later.
        with jt.flag_scope(use_cuda=self.device_flag):
            for dtype in SIGNED:
                for divisor in DIVISORS:
                    a, b, A, B = self._pair(
                        DIVIDENDS, [divisor] * len(DIVIDENDS), dtype)
                    quotient = (A // B).numpy()
                    remainder = jt.mod(A, B).numpy()
                    np.testing.assert_array_equal(
                        quotient * b + remainder, a,
                        err_msg="%s: (a // %d) * %d + a %% %d != a"
                                % (dtype, divisor, divisor, divisor))

    def test_the_operator_spelling_agrees_with_jt_mod(self):
        with jt.flag_scope(use_cuda=self.device_flag):
            a, b, A, B = self._pair(DIVIDENDS, [2] * len(DIVIDENDS), "int32")
            np.testing.assert_array_equal((A % B).numpy(), jt.mod(A, B).numpy())

    def test_unsigned_is_unchanged(self):
        # The floor adjustment keys off a negative remainder, which unsigned
        # arithmetic never produces. Pinned so a later rewrite cannot make
        # unsigned pay for the signed fix.
        with jt.flag_scope(use_cuda=self.device_flag):
            for dtype in UNSIGNED:
                a, b, A, B = self._pair((0, 1, 2, 7, 255), (2, 2, 3, 3, 4), dtype)
                np.testing.assert_array_equal(jt.mod(A, B).numpy(), a % b)

    def test_floats_still_floor(self):
        with jt.flag_scope(use_cuda=self.device_flag):
            for dtype in ("float32", "float64"):
                a = np.array([-7.0, -5.0, 5.0, 7.0], dtype=dtype)
                b = np.array([2.0, 3.0, -3.0, -2.0], dtype=dtype)
                got = jt.mod(jt.array(a), jt.array(b)).numpy()
                np.testing.assert_allclose(got, a % b, rtol=0, atol=1e-6)


class TestIntegerModFloorsCpu(_ModContract, unittest.TestCase):
    device_flag = 0


@unittest.skipUnless(_has_cuda(), "a CUDA device is required")
class TestIntegerModFloorsCuda(_ModContract, unittest.TestCase):
    device_flag = 1


if __name__ == "__main__":
    unittest.main()
