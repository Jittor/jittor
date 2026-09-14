# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""A float64 transcendental must be computed in float64 on every device.

The CUDA expression table spelled every unary maths function with its
**float-only** C variant -- `::logf`, `::expf`, `::sinf` and sixteen more --
whatever the operand's dtype. A float64 input was therefore narrowed to
float32, the function evaluated at single precision, and the result widened
back. `round` had been given a dtype dispatch at some point; the rest had not
(KI-OPS-007).

What made it hard to see is that most inputs hide it: `exp(1e-300)` is `1.0` in
both precisions. It shows where the answer lives below float32's resolution:

    log(1 + 2**-51)   CPU 4.4408920985006252e-16   CUDA 0.0

`1 + 2**-51` rounds to exactly `1.0` in float32, and `log(1.0)` is zero. A
gradient or a loss computed that way is not slightly off; it is gone.

The values here are chosen so the float32 answer is *qualitatively* wrong --
zero, or equal to the input -- rather than merely less precise. A tolerance
test would pass on the old build for several of them.

`auto_convert_64_to_32=0` is set because `jt.array` otherwise narrows 64-bit
input on construction (KI-DTYPE-002), which would make this file measure that
instead.
"""

from _helpers import capability as _test_capability

import unittest

import numpy as np

import jittor as jt


def _has_cuda():
    return bool(_test_capability.check_accelerator('cuda', backend=jt).enabled)


#: `1 + 2**-51` and `1e-17` are both exactly representable in float64 and both
#: collapse in float32 -- the first to `1.0`, the second to a value whose
#: transcendental is indistinguishable from the input.
JUST_ABOVE_ONE = 1.0 + 2.0 ** -51
TINY = 1e-17

#: `(name, jittor function, numpy reference, input)`. NumPy is a genuine
#: oracle here: these are IEEE-754 double-precision library functions, not a
#: matter of convention.
CASES = (
    ("log", jt.log, np.log, JUST_ABOVE_ONE),
    ("sqrt", jt.sqrt, np.sqrt, JUST_ABOVE_ONE),
    ("sin", jt.sin, np.sin, TINY),
    ("tan", jt.tan, np.tan, TINY),
    ("tanh", jt.tanh, np.tanh, TINY),
    ("arctan", jt.atan, np.arctan, TINY),
    ("arcsin", jt.asin, np.arcsin, TINY),
)


class _Float64UnaryMath:

    device_flag = 0

    def _evaluate(self, fn, value):
        with jt.flag_scope(use_cuda=self.device_flag, auto_convert_64_to_32=0):
            x = jt.array(np.array([value], dtype="float64"))
            self.assertEqual(str(x.dtype), "float64",
                             "the input was narrowed before the operator saw it")
            return float(np.asarray(fn(x).numpy()).ravel()[0])

    def test_float64_matches_numpy_to_the_bit(self):
        for name, fn, reference, value in CASES:
            with self.subTest(op=name):
                expected = float(reference(np.float64(value)))
                got = self._evaluate(fn, value)
                self.assertEqual(
                    got, expected,
                    "%s(%r) in float64 gave %r, NumPy gives %r" %
                    (name, value, got, expected))

    def test_the_inputs_really_do_collapse_in_float32(self):
        """Otherwise this file would be asserting nothing about narrowing."""
        self.assertEqual(np.float32(JUST_ABOVE_ONE), np.float32(1.0))
        self.assertEqual(float(np.log(np.float32(JUST_ABOVE_ONE))), 0.0)
        self.assertNotEqual(float(np.log(np.float64(JUST_ABOVE_ONE))), 0.0)

    def test_float32_is_unchanged(self):
        """The dispatch must not have moved the single-precision path.

        For a float32 operand the template emits the same `::logf` text it
        emitted before, so this is a check that the dispatch condition is the
        right way round -- an inverted one would compute float32 in double and
        show up here as an answer closer to NumPy's double result than
        `--use_fast_math` allows.
        """
        host = (np.random.RandomState(0).rand(1024).astype("float32") + 0.5)
        with jt.flag_scope(use_cuda=self.device_flag):
            got = np.asarray(jt.log(jt.array(host)).numpy())
        np.testing.assert_allclose(got, np.log(host), rtol=1e-5, atol=1e-6)


class TestFloat64UnaryMathCpu(_Float64UnaryMath, unittest.TestCase):
    device_flag = 0


@unittest.skipIf(not _has_cuda(), "no CUDA device")
class TestFloat64UnaryMathCuda(_Float64UnaryMath, unittest.TestCase):
    device_flag = 1


if __name__ == "__main__":
    unittest.main()
