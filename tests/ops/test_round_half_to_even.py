# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""``round`` breaks ties to even, and does not narrow a float64 to float32.

Two defects lived on the same two table rows in ``src/type/common_op_type.cc``.

1. ``std::round`` (CPU) and ``::roundf`` (CUDA) round halves *away from zero*.
   NumPy and Torch round halves *to even*: ``round(0.5) == 0``,
   ``round(2.5) == 2``, ``round(-0.5) == -0`` . Jittor answered 1, 3 and -1.

2. The CUDA row was the ``f`` -- i.e. *float* -- spelling. A float64 input was
   converted to float before rounding, so ``round(12345678901234.5)`` came back
   as 12345679020032.0 on a GPU and 12345678901234.0 on a CPU: the same call,
   two answers, decided by which device was present.

The half-to-even assertions do not lean on NumPy. Breaking ties to even *means*
the result is even, and that is what is asserted for every tie, so a later
rewrite that happened to match NumPy for these particular six numbers and not
in general still fails here. Likewise the precision assertion is an identity --
rounding a value that is already an integer must return that same value -- and
a float32 round trip cannot satisfy it above 2**24 no matter what NumPy says.

Dtypes are asserted rather than assumed: ``jt.array`` narrows 64-bit NumPy
values to 32 bits by default (KI-DTYPE-002), which would quietly turn the
float64 half of this file into a second float32 run.
"""

from _helpers import capability as _test_capability

import unittest

import numpy as np

import jittor as jt


def _has_cuda():
    return bool(_test_capability.check_accelerator('cuda', backend=jt).enabled)


#: Every tie in [-4.5, 4.5], both signs, including the two that surround zero.
TIES = np.array([-4.5, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5])

#: Integer-valued float64 values that no float32 can hold exactly. 2**24+1 is
#: the first integer float32 skips; the rest are further out.
BEYOND_FLOAT32 = np.array([
    2.0 ** 24 + 1.0,
    12345678901234.0,
    -12345678901234.0,
    4503599627370497.0,          # 2**52 + 1
    1e16 + 2.0,
], dtype="float64")


class _RoundContract:

    device_flag = 0

    def _var(self, values, dtype):
        with jt.flag_scope(auto_convert_64_to_32=0):
            v = jt.array(np.asarray(values, dtype=dtype))
        # KI-DTYPE-002: without the flag scope above this silently becomes
        # float32 and the float64 cases stop testing anything.
        self.assertEqual(str(v.dtype), dtype)
        return v

    def test_half_precision_ties_round_to_even_too(self):
        # float16/bfloat16 live in a separate table (src/type/fp16_op_type.cc).
        # Leaving them on half-away-from-zero would make jt.round answer
        # differently depending only on the input width.
        with jt.flag_scope(use_cuda=self.device_flag):
            for dtype in ("float16",):
                v = jt.array(TIES.astype(dtype))
                self.assertEqual(str(v.dtype), dtype)
                got = v.round().numpy().astype("float64")
                np.testing.assert_array_equal(
                    np.mod(got, 2.0), np.zeros_like(got),
                    err_msg="%s: round() left a tie on an odd integer: %r"
                            % (dtype, got.tolist()))

    def test_ties_round_to_an_even_integer(self):
        # The contract itself, stated without an oracle: a tie resolves to the
        # even neighbour. Rounding half away from zero puts 0.5 -> 1 and
        # 2.5 -> 3, which are odd.
        with jt.flag_scope(use_cuda=self.device_flag):
            for dtype in ("float32", "float64"):
                got = self._var(TIES, dtype).round().numpy()
                np.testing.assert_array_equal(
                    np.mod(got, 2.0), np.zeros_like(got),
                    err_msg="%s: round() left a tie on an odd integer: %r"
                            % (dtype, got.tolist()))
                np.testing.assert_array_equal(
                    np.abs(got - TIES.astype(dtype)),
                    np.full(TIES.shape, 0.5, dtype=dtype),
                    err_msg="%s: a tie moved by something other than 0.5" % dtype)

    def test_ties_match_numpy(self):
        with jt.flag_scope(use_cuda=self.device_flag):
            for dtype in ("float32", "float64"):
                raw = TIES.astype(dtype)
                got = self._var(TIES, dtype).round().numpy()
                np.testing.assert_array_equal(got, np.round(raw),
                                              err_msg="%s tie handling" % dtype)
                # -0.5 rounds to -0.0, not to +0.0; the sign is observable.
                np.testing.assert_array_equal(np.signbit(got),
                                              np.signbit(np.round(raw)),
                                              err_msg="%s tie sign" % dtype)

    def test_non_ties_are_unchanged(self):
        with jt.flag_scope(use_cuda=self.device_flag):
            raw = np.array([-2.7, -2.2, -0.4, 0.4, 1.2, 1.8, 7.0, -7.0])
            for dtype in ("float32", "float64"):
                got = self._var(raw, dtype).round().numpy()
                np.testing.assert_array_equal(got, np.round(raw.astype(dtype)),
                                              err_msg="%s non-tie" % dtype)

    def test_rounding_an_integer_returns_it_unchanged(self):
        # An identity, so it holds for every implementation that is not
        # secretly narrowing: these float64 values are already integers, and
        # every one of them is outside float32's exactly-representable range.
        with jt.flag_scope(use_cuda=self.device_flag):
            v = self._var(BEYOND_FLOAT32, "float64")
            got = v.round().numpy()
            np.testing.assert_array_equal(
                got, BEYOND_FLOAT32,
                err_msg="round() narrowed a float64 to float32: %r" % got.tolist())

    def test_float64_ties_keep_full_precision(self):
        with jt.flag_scope(use_cuda=self.device_flag):
            raw = np.array([12345678901234.5, -12345678901234.5], dtype="float64")
            got = self._var(raw, "float64").round().numpy()
            np.testing.assert_array_equal(got, np.round(raw))

    def test_dtype_is_preserved(self):
        with jt.flag_scope(use_cuda=self.device_flag):
            for dtype in ("float32", "float64"):
                self.assertEqual(str(self._var(TIES, dtype).round().dtype), dtype)


class TestRoundHalfToEvenCpu(_RoundContract, unittest.TestCase):
    device_flag = 0


@unittest.skipUnless(_has_cuda(), "a CUDA device is required")
class TestRoundHalfToEvenCuda(_RoundContract, unittest.TestCase):
    device_flag = 1


if __name__ == "__main__":
    unittest.main()
