# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""float64 unary math keeps float64 precision -- CPU yes, CUDA mostly no.

Almost every row of the CUDA table in ``src/type/common_op_type.cc`` is the
``f`` -- i.e. *float* -- spelling of its libm function: ``::floorf``,
``::ceilf``, ``::sqrtf``, ``::expf``, ``::logf``, ``::sinf`` and the rest. A
float64 operand is converted to float on the way in, so the result carries 24
bits of mantissa instead of 53. ``ceil(12345678901234.5)`` is not merely
imprecise there, it is wrong: 12345679020032.0 instead of 12345678901235.0.
The CPU table uses the unsuffixed overloads and is exact.

``round`` used to be in that list and is now dispatched on width, which is what
the exact assertions below cover; the strict expected failure covers the rows
that were left alone (KI-OPS-007). Both halves live here so that fixing the
rest of the table turns the xfail red against a file that already states what
"fixed" looks like.

The reference values are chosen beyond 2**24, where float32 cannot represent
consecutive integers -- inside that range the two widths agree and nothing is
observable.
"""

from _helpers import capability as _test_capability

import unittest

import numpy as np
import pytest

import jittor as jt


def _has_cuda():
    return bool(_test_capability.check_accelerator('cuda', backend=jt).enabled)


#: Integer-valued and half-integer float64 values that float32 cannot hold.
WIDE = np.array([12345678901234.5, 2.0 ** 24 + 1.0, 1.0000000000000002],
                dtype="float64")


def _f64(values):
    with jt.flag_scope(auto_convert_64_to_32=0):
        # KI-DTYPE-002: jt.array narrows 64-bit numpy values by default.
        v = jt.array(np.asarray(values, dtype="float64"))
    assert str(v.dtype) == "float64", v.dtype
    return v


class _Float64UnaryPrecision:

    device_flag = 0

    def test_round_keeps_float64_precision(self):
        with jt.flag_scope(use_cuda=self.device_flag):
            got = _f64(WIDE).round().numpy()
            np.testing.assert_array_equal(got, np.round(WIDE))

    def _check_family(self):
        with jt.flag_scope(use_cuda=self.device_flag):
            positive = np.abs(WIDE)
            for name, npf, src in (("floor", np.floor, WIDE),
                                   ("ceil", np.ceil, WIDE),
                                   ("sqrt", np.sqrt, positive),
                                   ("log", np.log, positive)):
                got = getattr(_f64(src), name)().numpy()
                np.testing.assert_array_equal(
                    got, npf(src),
                    err_msg="%s narrowed float64 to float32" % name)


class TestFloat64UnaryPrecisionCpu(_Float64UnaryPrecision, unittest.TestCase):
    device_flag = 0

    def test_unary_family_keeps_float64_precision(self):
        self._check_family()


@unittest.skipUnless(_has_cuda(), "a CUDA device is required")
class TestFloat64UnaryPrecisionCuda(_Float64UnaryPrecision, unittest.TestCase):
    device_flag = 1

    @pytest.mark.xfail(strict=True,
                       reason="KI-OPS-007: CUDA table uses float-only libm spellings")
    def test_unary_family_keeps_float64_precision(self):
        self._check_family()


if __name__ == "__main__":
    unittest.main()
