# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Kernels must not be compiled under a promise that NaN does not occur.

``-Ofast`` implies ``-ffast-math``, which implies ``-ffinite-math-only``: the
compiler is told no operand is ever infinite or NaN and optimises on it. Under
that flag ``1 / 0`` came back as ``nan`` instead of ``inf`` and a hand-written
``x != x`` may be folded to false, so the arithmetic *and* the checks written
to catch it both stop working (KI-BACKEND-005).

The gate that matters is behavioural -- ``tests/ops/test_ieee_arithmetic.py``
evaluates the IEEE table on both devices, and it goes red the moment the flag
comes back. This file is the cheaper companion: it names the flag, so the
failure says *what* was changed rather than only that arithmetic broke, and it
covers the flag being reintroduced somewhere the IEEE table happens not to
reach.

What this file does **not** establish: that no fast-math is in effect anywhere.
It reads the flags the build settled on, so a flag injected through the
environment or a backend's own compile line is outside it. That is why the
behavioural test is the gate and this one is the label.
"""

import unittest

import jittor as jt


#: Flags that grant the compiler a finite-math assumption, and the ones that
#: imply them. `-Ofast` is the one this project actually shipped.
FORBIDDEN = ("-Ofast", "-ffast-math", "-ffinite-math-only", "--use_fast_math")


class TestKernelMathFlags(unittest.TestCase):

    def test_no_finite_math_promise_in_the_cpu_kernel_flags(self):
        flags = jt.flags.cc_flags
        self.assertTrue(flags.strip(), "cc_flags is empty; this test would "
                                       "pass without checking anything")
        for flag in FORBIDDEN:
            with self.subTest(flag=flag):
                self.assertNotIn(
                    flag, flags,
                    "%s promises the compiler that infinities and NaN do not "
                    "occur. Under it `1/0` returns nan and `x != x` may be "
                    "folded to false (KI-BACKEND-005). If this is deliberate, "
                    "the IEEE table in tests/ops/test_ieee_arithmetic.py has "
                    "to be reconciled first." % flag)

    def test_an_optimisation_level_is_still_being_asked_for(self):
        """The fix was to swap the flag, not to drop it.

        Removing `-Ofast` and putting nothing back would leave kernels at the
        compiler's default `-O0` and pass the test above, which is the way this
        particular gate can be satisfied by making things worse.
        """
        self.assertRegex(
            jt.flags.cc_flags, r"-O[123s]\b",
            "kernel flags carry no optimisation level at all")


if __name__ == "__main__":
    unittest.main()
