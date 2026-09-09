# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""max/min drop NaN where sum/mean/prod propagate it -- KI-OPS-006.

NumPy and Torch propagate NaN through ``maximum``/``minimum`` and through the
``max()``/``min()`` reductions built on them. Jittor does not::

    jt.max([nan, 1.0, 2.0]) -> 2.0      numpy -> nan
    jt.min([nan, 1.0, 2.0]) -> 1.0      numpy -> nan

``std::max(a, b)`` is ``a < b ? b : a`` and CUDA's ``::max`` on floats lowers
to ``fmaxf``; every comparison against NaN is false, so the operand that is
*not* NaN survives, and a reduction that starts at -inf and folds
``max(acc, x)`` can never report one. ``sum``/``mean``/``prod`` were never
affected: IEEE addition and multiplication propagate NaN in hardware, with no
comparison involved. Those are asserted here as the control -- they are what
makes this a *divergence between reductions* rather than a global policy.

The elementwise half diverges from itself as well: ``maximum(1.0, nan)`` is
1.0 on CPU and ``maximum(nan, 1.0)`` is 1.0 on CUDA, because the two backends
lower the ternary differently.

The expected failures are strict, so the day the operator is fixed these turn
red and KI-OPS-006 has to be retired rather than outliving the defect.
"""

from _helpers import capability as _test_capability

import unittest

import numpy as np
import pytest

import jittor as jt


def _has_cuda():
    return bool(_test_capability.check_accelerator('cuda', backend=jt).enabled)


#: One NaN in the middle, so neither a first-element nor a last-element special
#: case can hide the answer.
WITH_NAN = np.array([1.0, np.nan, 2.0, -3.0], dtype="float32")


class _NanPropagationContract:

    device_flag = 0

    def test_sum_mean_and_prod_propagate_nan(self):
        # The control: these reductions are correct, which is what makes
        # max/min a divergence inside one operator family.
        with jt.flag_scope(use_cuda=self.device_flag):
            x = jt.array(WITH_NAN)
            self.assertEqual(str(x.dtype), "float32")
            self.assertTrue(np.isnan(float(x.sum().numpy())))
            self.assertTrue(np.isnan(float(x.mean().numpy())))
            self.assertTrue(np.isnan(float(jt.prod(x).numpy())))

    def test_max_and_min_without_nan_are_correct(self):
        with jt.flag_scope(use_cuda=self.device_flag):
            raw = np.array([1.0, 5.0, 2.0, -3.0], dtype="float32")
            x = jt.array(raw)
            self.assertEqual(float(x.max().numpy()), float(raw.max()))
            self.assertEqual(float(x.min().numpy()), float(raw.min()))

    @pytest.mark.xfail(strict=True,
                       reason="KI-OPS-006: max/min reductions drop NaN")
    def test_max_and_min_reductions_propagate_nan(self):
        with jt.flag_scope(use_cuda=self.device_flag):
            x = jt.array(WITH_NAN)
            self.assertEqual(str(x.dtype), "float32")
            self.assertTrue(np.isnan(float(x.max().numpy())),
                            "max() dropped the NaN")
            self.assertTrue(np.isnan(float(x.min().numpy())),
                            "min() dropped the NaN")

    @pytest.mark.xfail(strict=True,
                       reason="KI-OPS-006: elementwise maximum/minimum drop NaN")
    def test_elementwise_maximum_and_minimum_propagate_nan(self):
        with jt.flag_scope(use_cuda=self.device_flag):
            a = np.array([np.nan, 1.0, np.nan], dtype="float32")
            b = np.array([1.0, np.nan, np.nan], dtype="float32")
            got_max = jt.maximum(jt.array(a), jt.array(b)).numpy()
            got_min = jt.minimum(jt.array(a), jt.array(b)).numpy()
            np.testing.assert_array_equal(np.isnan(got_max), np.isnan(np.maximum(a, b)))
            np.testing.assert_array_equal(np.isnan(got_min), np.isnan(np.minimum(a, b)))


class TestMinMaxNanPropagationCpu(_NanPropagationContract, unittest.TestCase):
    device_flag = 0


@unittest.skipUnless(_has_cuda(), "a CUDA device is required")
class TestMinMaxNanPropagationCuda(_NanPropagationContract, unittest.TestCase):
    device_flag = 1


if __name__ == "__main__":
    unittest.main()
