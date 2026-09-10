# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""``abs`` has three gradient cases, and zero is the one that was missing.

The backward was `x >= 0 ? dout : -dout`, which puts zero in the positive
branch and hands it a gradient of 1. Torch and the minimum-norm subgradient
both give 0 there.

It is not only a convention. An L1 penalty is chosen precisely because it holds
weights at exactly zero; a gradient of 1 at zero pushes them off it, so the
term stops doing the thing it was added for. Negative zero took the same wrong
branch, because `-0.0 >= 0` is true.

The two ordinary sides are asserted alongside, so a later rewrite of the
three-way selection cannot fix zero by breaking the sign of everything else.
"""

from _helpers import capability as _test_capability

import unittest

import numpy as np

import jittor as jt


def _has_cuda():
    return bool(_test_capability.check_accelerator('cuda', backend=jt).enabled)


class _AbsGradientContract:

    device_flag = 0

    def _grad(self, values):
        with jt.flag_scope(use_cuda=self.device_flag):
            x = jt.array(np.array(values, dtype="float32"))
            return jt.grad(jt.abs(x), x).numpy()

    def test_zero_gets_no_gradient(self):
        np.testing.assert_array_equal(self._grad([0.0]), [0.0])

    def test_negative_zero_gets_no_gradient(self):
        # `-0.0 >= 0` is true, so the old two-way form sent it to the positive
        # branch. Kept separate from positive zero because they reach the bug
        # by different routes.
        np.testing.assert_array_equal(self._grad([-0.0]), [0.0])

    def test_the_two_ordinary_sides_are_unchanged(self):
        np.testing.assert_array_equal(self._grad([-2.0, 2.0]), [-1.0, 1.0])

    def test_all_three_cases_together(self):
        np.testing.assert_array_equal(
            self._grad([-2.0, -0.0, 0.0, 2.0]), [-1.0, 0.0, 0.0, 1.0])

    def test_an_l1_penalty_leaves_a_weight_sitting_at_zero(self):
        # The reason this matters, stated as the behaviour rather than the
        # number: L1 is used to hold weights at zero, and the defect pushed
        # them off.
        with jt.flag_scope(use_cuda=self.device_flag):
            w = jt.array(np.array([0.0, 0.5], dtype="float32"))
            g = jt.grad(jt.abs(w).sum(), w).numpy()
            self.assertEqual(float(g[0]), 0.0,
                             "an L1 term pushed a weight away from exactly zero")


class TestAbsGradientAtZeroCpu(_AbsGradientContract, unittest.TestCase):
    device_flag = 0


@unittest.skipUnless(_has_cuda(), "a CUDA device is required")
class TestAbsGradientAtZeroCuda(_AbsGradientContract, unittest.TestCase):
    device_flag = 1


if __name__ == "__main__":
    unittest.main()
