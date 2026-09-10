# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Arithmetic on infinities and NaN must follow IEEE-754, on every device.

CPU kernels were built with ``-Ofast``, which implies ``-ffast-math``, which
implies ``-ffinite-math-only`` -- a promise to the compiler that no operand is
ever infinite or NaN. It then optimised on that promise, and operands that
*were* infinite took whatever path the transformed code happened to produce::

    1 / 0        gave nan   should be inf
    -inf / 0     gave nan   should be -inf

A single element computed correctly; the wrong answers began at length 4, which
is where the kernel vectorises, so a scalar spot-check saw nothing. CUDA was
correct throughout, which is the other half of what makes this expensive: the
same expression on the same data disagreed between the two devices
(KI-BACKEND-005).

The dangerous shape is not the obviously broken one. A fully masked attention
row subtracts its own ``-inf`` maximum; a finite result there produces a
well-formed but wrong softmax rather than an obvious ``nan``, and nothing in the
run says anything happened.

Every case is a value IEEE-754 defines exactly, so NumPy is a genuine oracle
rather than a second opinion -- the answers are not a matter of convention.
Length 8 is deliberate: at length 1 all of these passed even when the flag was
wrong.
"""

from _helpers import capability as _test_capability

import unittest

import numpy as np

import jittor as jt


def _has_cuda():
    return bool(_test_capability.check_accelerator('cuda', backend=jt).enabled)


#: Long enough to vectorise. The defect was invisible below 4.
N = 8

INF = float("inf")
NAN = float("nan")

#: ``(name, left, right, operator, expected)``. Written as operands rather than
#: as expressions so the same table runs on both devices unchanged.
CASES = (
    ("inf - inf", INF, INF, lambda a, b: a - b, NAN),
    ("1 / 0", 1.0, 0.0, lambda a, b: a / b, INF),
    ("-1 / 0", -1.0, 0.0, lambda a, b: a / b, -INF),
    ("-inf / 0", -INF, 0.0, lambda a, b: a / b, -INF),
    ("inf / inf", INF, INF, lambda a, b: a / b, NAN),
    ("inf * 1", INF, 1.0, lambda a, b: a * b, INF),
    ("0 * inf", 0.0, INF, lambda a, b: a * b, NAN),
    ("inf + -inf", INF, -INF, lambda a, b: a + b, NAN),
    ("nan + 1", NAN, 1.0, lambda a, b: a + b, NAN),
    ("nan * 0", NAN, 0.0, lambda a, b: a * b, NAN),
)


class _IeeeArithmetic:

    device_flag = 0

    def _evaluate(self, left, right, op):
        with jt.flag_scope(use_cuda=self.device_flag):
            a = jt.array(np.full(N, left, dtype="float32"))
            b = jt.array(np.full(N, right, dtype="float32"))
            # KI-DTYPE-002: jt.array narrows 64-bit values. Say out loud which
            # width this is measuring, so a narrowing cannot be read as the
            # arithmetic being wrong.
            assert str(a.dtype) == "float32", a.dtype
            return np.asarray(op(a, b).numpy(), dtype=np.float64)

    def test_ieee_special_values(self):
        for name, left, right, op, expected in CASES:
            with self.subTest(case=name):
                got = self._evaluate(left, right, op)
                self.assertEqual(got.shape, (N,))
                # Compared elementwise: the defect appeared only in the
                # vectorised tail on some builds, so checking got[0] alone
                # would reproduce the blind spot this file exists to close.
                if np.isnan(expected):
                    self.assertTrue(
                        np.all(np.isnan(got)),
                        "%s gave %s, IEEE says nan" % (name, got))
                else:
                    np.testing.assert_array_equal(
                        got, np.full(N, expected),
                        err_msg="%s gave %s, IEEE says %s" % (name, got, expected))

    def test_numpy_agrees_on_every_case(self):
        """The oracle, run on the same operands rather than trusted from memory.

        If NumPy ever disagreed with the table above, the table would be the
        thing that is wrong, and this says so instead of leaving a stale
        expectation asserted against the code.
        """
        for name, left, right, op, expected in CASES:
            with self.subTest(case=name):
                a = np.full(N, left, dtype="float32")
                b = np.full(N, right, dtype="float32")
                with np.errstate(all="ignore"):
                    reference = np.asarray(op(a, b), dtype=np.float64)
                if np.isnan(expected):
                    self.assertTrue(np.all(np.isnan(reference)), name)
                else:
                    np.testing.assert_array_equal(
                        reference, np.full(N, expected), err_msg=name)

    def test_a_nan_test_written_as_a_comparison_still_works(self):
        """``x != x`` has to survive the optimiser.

        Under ``-ffinite-math-only`` the compiler is free to fold this to
        false, and then every hand-written NaN check in the codebase silently
        stops checking. This is the predicate, not the arithmetic, and it is
        the reason ``nan_checker`` needed its own compile flags.
        """
        with jt.flag_scope(use_cuda=self.device_flag):
            x = jt.array(np.full(N, NAN, dtype="float32"))
            np.testing.assert_array_equal(
                (x != x).numpy(), np.ones(N, dtype="bool"))
            y = jt.array(np.arange(N, dtype="float32"))
            np.testing.assert_array_equal(
                (y != y).numpy(), np.zeros(N, dtype="bool"))


class TestIeeeArithmeticCpu(_IeeeArithmetic, unittest.TestCase):
    device_flag = 0


@unittest.skipIf(not _has_cuda(), "no CUDA device")
class TestIeeeArithmeticCuda(_IeeeArithmetic, unittest.TestCase):
    device_flag = 1


if __name__ == "__main__":
    unittest.main()
