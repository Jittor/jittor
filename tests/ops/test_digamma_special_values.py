# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""``digamma`` at the arguments where it has a pole or is undefined.

The implementation writes these three answers out by hand, following the C++
standard's gamma-family wording and SciPy: a negative integer gives NaN, a
signed zero gives the infinity of the opposite sign, and a NaN propagates.
Every one of those is a branch, and two of the three used to be compiled away.

Under ``-Ofast`` the CPU kernel answered ``-inf`` for both ``nan`` and ``-0.0``
(KI-OPS-011, recorded 2026-09-10 against ``scipy.special.digamma``). The cause
was not in this function at all: ``-ffast-math`` implies
``-ffinite-math-only``, under which ``x == 0`` folds so that ``-0.0`` takes the
``copysign(INFINITY, -x)`` path with the wrong sign, and the NaN branch --
reached only through comparisons that the same promise lets the compiler
decide statically -- was dropped. Building CPU kernels at ``-O3``
(KI-BACKEND-005, commit 1e50d76c5) removed the promise and both answers came
back, which is why this file exists on the other side of that change rather
than as part of it.

The ordinary values are here too. A pole is easy to pin by special-casing it,
and a special case that quietly damaged the series would then pass unnoticed;
these say the main branch still computes digamma.
"""

import unittest

import numpy as np

import jittor as jt

from _helpers import capability as _test_capability


def _has_cuda():
    return bool(_test_capability.check_accelerator('cuda', backend=jt).enabled)


INF = float("inf")
NAN = float("nan")

#: ``(name, input, expected)`` -- the arguments where digamma is a pole or is
#: undefined. Values are what SciPy and the C++ standard give, cross-checked
#: against SciPy by :meth:`test_scipy_agrees_on_every_special_value` rather
#: than trusted from memory.
SPECIAL = (
    ("nan propagates", NAN, NAN),
    ("negative zero", -0.0, INF),
    ("positive zero", 0.0, -INF),
    ("positive infinity", INF, INF),
    ("negative infinity", -INF, NAN),
    ("negative integer -1", -1.0, NAN),
    ("negative integer -3", -3.0, NAN),
)

#: Arguments away from any pole, to say the series itself still works.
ORDINARY = (-0.5, 0.5, 1.0, 2.5, 10.0, 37.25)


class _DigammaSpecialValues:

    device_flag = 0

    def _digamma(self, values):
        """Evaluate on one array, on this class's device, and say where it ran.

        All the special arguments go through in a single array rather than one
        call each: that is the shape a vectorised kernel sees, and the defect
        this file pins lived in the vectorised path of a neighbouring operator.
        """
        with jt.flag_scope(use_cuda=self.device_flag):
            x = jt.array(np.asarray(values, dtype="float32"))
            # KI-DTYPE-002: jt.array narrows 64-bit input. Say which width is
            # under test so a narrowing cannot be read as digamma being wrong.
            assert str(x.dtype) == "float32", x.dtype
            y = jt.digamma.apply(x)
            y.sync()
            expected_location = "device" if self.device_flag else "cpu"
            assert y.location() == expected_location, (
                "ran on %s, meant to measure %s"
                % (y.location(), expected_location))
            return np.asarray(y.numpy(), dtype=np.float64)

    def test_special_values(self):
        got = self._digamma([value for _, value, _ in SPECIAL])
        for (name, value, expected), answer in zip(SPECIAL, got):
            with self.subTest(case=name):
                if np.isnan(expected):
                    self.assertTrue(
                        np.isnan(answer),
                        "digamma(%r) gave %r, should be nan" % (value, answer))
                else:
                    self.assertEqual(
                        answer, expected,
                        "digamma(%r) gave %r, should be %r"
                        % (value, answer, expected))

    def test_ordinary_values_still_computed(self):
        from scipy import special

        got = self._digamma(ORDINARY)
        reference = special.digamma(np.asarray(ORDINARY, dtype=np.float64))
        # float32 through a reflection formula and an asymptotic series: a few
        # ulps, not bit equality. Tight enough that a broken branch shows.
        np.testing.assert_allclose(got, reference, rtol=2e-5, atol=1e-6)

    def test_scipy_agrees_on_every_special_value(self):
        """The oracle, run rather than remembered.

        If SciPy ever disagreed with the table above, the table would be the
        thing that is wrong, and this says so instead of leaving a stale
        expectation asserted against the kernel.
        """
        from scipy import special

        for name, value, expected in SPECIAL:
            with self.subTest(case=name):
                with np.errstate(all="ignore"):
                    reference = float(special.digamma(np.float64(value)))
                if np.isnan(expected):
                    self.assertTrue(np.isnan(reference), name)
                else:
                    self.assertEqual(reference, expected, name)


class TestDigammaSpecialValuesCpu(_DigammaSpecialValues, unittest.TestCase):
    device_flag = 0


@unittest.skipIf(not _has_cuda(), "no CUDA device")
class TestDigammaSpecialValuesCuda(_DigammaSpecialValues, unittest.TestCase):
    device_flag = 1


if __name__ == "__main__":
    unittest.main()
