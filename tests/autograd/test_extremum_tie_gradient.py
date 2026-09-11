# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""The gradient of a max/min reduction must sum to one, ties or not.

Every element equal to the extremum lands on the backward's mask, and each of
them used to receive the whole cotangent: ``max([1,3,3])`` produced ``[0,1,1]``,
summing to 2, and three ties summed to 3. The gradient was multiplied by the
number of tied extrema.

This is not a tie-breaking preference. ``max`` selects one input and is
1-homogeneous in it, so any subgradient sums to 1 whichever element it favours;
torch's ``max(dim)`` routes it all to the argmax and ``amax`` splits it evenly,
and both sum to 1. Jittor's reduction returns values rather than a
``(values, indices)`` pair, so the even split is the matching convention.

The forward value was always right, which is why nothing noticed: only the
gradient was wrong, and only on inputs where two elements are exactly equal --
which real data reaches constantly through relu outputs, padding and quantised
values.

Two neighbours are asserted as *correct* here rather than assumed: the binary
``maximum(a, b)`` already splits properly when its arguments are equal, and
``max_pool2d`` already routes to a single element. A fix aimed at the reduction
must not disturb either.
"""

from _helpers import capability as _test_capability

import unittest

import numpy as np

import jittor as jt


#: Probed when the case runs, not when the file is imported.
#:
#: ``unittest.skipUnless(_has_cuda(), ...)`` evaluated the probe in a decorator
#: argument, which runs at *collection* -- where this suite forbids backend
#: work -- and froze one answer for the whole process.
requires_cuda = _test_capability.accelerator_required('cuda', backend=jt)


class _TieContract:

    device_flag = 0

    def _grad_sum(self, fn, values):
        with jt.flag_scope(use_cuda=self.device_flag):
            x = jt.array(np.array(values, dtype="float32"))
            return jt.grad(fn(x), x).numpy()

    def test_max_without_ties_sums_to_one(self):
        g = self._grad_sum(jt.max, [1.0, 2.0, 3.0])
        self.assertAlmostEqual(float(g.sum()), 1.0, places=5)
        np.testing.assert_allclose(g, [0.0, 0.0, 1.0], atol=1e-6)

    def test_max_with_ties_still_sums_to_one(self):
        for values, ties in (([1.0, 3.0, 3.0], 2), ([3.0, 3.0, 3.0], 3),
                             ([3.0, 3.0, 3.0, 3.0], 4)):
            g = self._grad_sum(jt.max, values)
            self.assertAlmostEqual(
                float(g.sum()), 1.0, places=5,
                msg="%d tied maxima inflated the gradient to %.3f"
                    % (ties, g.sum()))

    def test_min_with_ties_still_sums_to_one(self):
        g = self._grad_sum(jt.min, [1.0, 1.0, 3.0])
        self.assertAlmostEqual(float(g.sum()), 1.0, places=5)

    def test_the_share_is_equal_among_ties(self):
        # Which rule is implemented is part of the contract, not an accident:
        # a later change to first-wins would keep the sum at 1 and silently
        # move where the gradient lands.
        g = self._grad_sum(jt.max, [1.0, 3.0, 3.0])
        np.testing.assert_allclose(g, [0.0, 0.5, 0.5], atol=1e-6)

    def test_each_reduced_row_sums_to_one(self):
        with jt.flag_scope(use_cuda=self.device_flag):
            a = np.array([[3.0, 3.0], [1.0, 2.0]], dtype="float32")
            x = jt.array(a)
            g = jt.grad(jt.max(x, 1).sum(), x).numpy()
            np.testing.assert_allclose(g.sum(1), [1.0, 1.0], atol=1e-6)

    def test_the_binary_maximum_is_unaffected(self):
        # Already correct before the reduction was fixed; asserted so the fix
        # cannot regress it.
        with jt.flag_scope(use_cuda=self.device_flag):
            a = jt.array(np.array([2.0, 2.0], dtype="float32"))
            b = jt.array(np.array([2.0, 1.0], dtype="float32"))
            ga = jt.grad(jt.maximum(a, b).sum(), a).numpy()
            gb = jt.grad(jt.maximum(a, b).sum(), b).numpy()
            np.testing.assert_allclose(ga + gb, [1.0, 1.0], atol=1e-6)

    def test_max_pool_is_unaffected(self):
        with jt.flag_scope(use_cuda=self.device_flag):
            x = jt.array(np.zeros((1, 1, 2, 2), dtype="float32"))
            g = jt.grad(jt.nn.max_pool2d(x, 2).sum(), x).numpy()
            self.assertAlmostEqual(float(g.sum()), 1.0, places=5)


class TestExtremumTieGradientCpu(_TieContract, unittest.TestCase):
    device_flag = 0


@requires_cuda
class TestExtremumTieGradientCuda(_TieContract, unittest.TestCase):
    device_flag = 1


if __name__ == "__main__":
    unittest.main()
