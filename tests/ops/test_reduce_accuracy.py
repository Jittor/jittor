# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""A float32 reduction's error must not grow with the number of elements.

A single running total rounds against a sum that is by then much larger than
the next element, so its error is proportional to ``n``: the CPU ``sum`` of
sixteen million copies of ``0.1`` was 15% wrong, growing 6.2e-4 -> 9.9e-3 ->
1.5e-1 as the size went 65,536 -> 1,048,576 -> 16,777,216, while CUDA's tree
reduction stayed near 1.6e-7 throughout (KI-BACKEND-006). Nothing warned; the
answer was a plausible number rather than an obviously broken one.

What is asserted here is the *shape of the growth*, not a threshold. A fixed
tolerance would have to be loose enough for the widest machine and compiler in
the fleet, and would then turn red or green for reasons that have nothing to do
with this defect. The two properties below do not depend on the host:

* the relative error at 16M stays within a small factor of the error at
  65,536 -- a serial accumulator gives 248x, a blocked one gives 1x;
* it stays within a small factor of what NumPy's pairwise summation, the
  reference shape, produces on the same input.

Every element is the same value, so the roundings all point the same way. That
is the worst case, and it is the one that made the defect visible: on random
input the errors cancel and a broken reduction still looks passable.

Both devices run. CUDA was already right, and the point of a regression test
here is that it stays right: a fix written against the CPU table alone, with
the accelerator left on its old behaviour, is a mistake this suite has caught
before.
"""

from _helpers import capability as _test_capability

import unittest

import numpy as np

import jittor as jt


def _has_cuda():
    return bool(_test_capability.check_accelerator('cuda', backend=jt).enabled)


#: float32 machine epsilon. No reduction can be asked to do better, so it is
#: the floor every ratio below is taken against.
EPS = float(np.finfo(np.float32).eps)

#: 65,536 to 16,777,216 -- a 256x range. A serial accumulator's error grows by
#: about that factor across it; anything blocked or tree-shaped does not.
SIZES = (1 << 16, 1 << 20, 1 << 24)

#: How much the relative error may grow across that 256x range, and how far it
#: may sit above NumPy's. Far below the 248x a serial accumulator gives and far
#: above the 1x-4x a blocked one does, so this discriminates the defect rather
#: than the host.
SPREAD = 10.0

#: The value summed. 0.1 is not exact in binary, so every partial sum rounds.
VALUE = np.float32(0.1)


class _ReductionErrorDoesNotGrow:

    device_flag = 0

    def _errors(self, reduce, exact_of, shape_of):
        """Relative error against the exact result, at each size."""
        out = []
        for n in SIZES:
            host = np.full(shape_of(n), VALUE, dtype="float32")
            with jt.flag_scope(use_cuda=self.device_flag):
                x = jt.array(host)
                # KI-DTYPE-002: jt.array narrows 64-bit values. Say out loud
                # which width this test is measuring.
                assert str(x.dtype) == "float32", x.dtype
                value = float(reduce(x).item())
                del x
                jt.gc()
            exact = exact_of(n)
            out.append(abs(value - exact) / abs(exact))
        return out

    def _assert_flat(self, errors, what):
        floor = max(min(errors), EPS)
        self.assertLessEqual(
            max(errors), SPREAD * floor,
            "%s relative error grows with size: %s over n=%s. An error "
            "proportional to n is the signature of a single serial "
            "accumulator (KI-BACKEND-006)."
            % (what, ["%.3g" % e for e in errors], list(SIZES)))

    def test_sum_error_does_not_grow_with_size(self):
        self._assert_flat(
            self._errors(jt.sum, lambda n: float(VALUE) * n, lambda n: (n,)),
            "sum")

    def test_mean_error_does_not_grow_with_size(self):
        self._assert_flat(
            self._errors(jt.mean, lambda n: float(VALUE), lambda n: (n,)),
            "mean")

    def test_nested_sum_error_does_not_grow_with_size(self):
        """The same, over a reduction that is a loop nest rather than a loop.

        ``x.sum()`` on a batch of images reduces three dimensions at once, and
        blocking only the innermost of them would leave the outer two chained:
        the same linear growth, divided by the innermost trip count. This is
        the shape that catches a fix applied one loop deep.
        """
        self._assert_flat(
            self._errors(jt.sum, lambda n: float(VALUE) * n,
                         lambda n: (n >> 12, 1 << 6, 1 << 6)),
            "nested sum")

    def test_sum_error_is_near_numpy_at_sixteen_million(self):
        n = SIZES[-1]
        host = np.full(n, VALUE, dtype="float32")
        exact = float(VALUE) * n
        with jt.flag_scope(use_cuda=self.device_flag):
            x = jt.array(host)
            assert str(x.dtype) == "float32", x.dtype
            got = abs(float(jt.sum(x).item()) - exact) / exact
            del x
            jt.gc()
        reference = abs(float(host.sum()) - exact) / exact
        self.assertLessEqual(
            got, SPREAD * max(reference, EPS),
            "sum of %d float32 elements is %.3g off, against %.3g for NumPy's "
            "pairwise summation on the same input (KI-BACKEND-006)."
            % (n, got, reference))


class TestReductionErrorDoesNotGrowCpu(_ReductionErrorDoesNotGrow,
                                       unittest.TestCase):
    device_flag = 0


@unittest.skipUnless(_has_cuda(), "a CUDA device is required")
class TestReductionErrorDoesNotGrowCuda(_ReductionErrorDoesNotGrow,
                                        unittest.TestCase):
    device_flag = 1


@unittest.skipUnless(_has_cuda(), "a CUDA device is required")
class TestLargeReductionAgreesAcrossDevices(unittest.TestCase):
    """The parity case the ledger entry asked for.

    ``tests/backends/parity`` compares CPU against the accelerator, which is
    exactly the comparison that would have shown this, and missed it because
    every case there is small enough for a serial accumulator to still look
    right. This one is not: at sixteen million elements the two devices
    disagreed by 15%, and the wrong side was the one used as the reference.
    """

    #: Both devices land within a few ulps of the exact value, so their
    #: difference is bounded by their own accuracy with room to spare. The
    #: defect this guards produced 0.15, five orders of magnitude above it.
    AGREEMENT = 100 * EPS

    def test_sixteen_million_element_sum_agrees(self):
        n = 1 << 24
        host = np.full(n, VALUE, dtype="float32")
        values = {}
        for name, flag in (("cpu", 0), ("cuda", 1)):
            with jt.flag_scope(use_cuda=flag):
                x = jt.array(host)
                assert str(x.dtype) == "float32", x.dtype
                values[name] = float(jt.sum(x).item())
                del x
                jt.gc()
        spread = abs(values["cpu"] - values["cuda"]) / abs(values["cuda"])
        self.assertLessEqual(
            spread, self.AGREEMENT,
            "CPU and CUDA disagree by %.3g on the sum of %d float32 elements: "
            "%r against %r. A backend comparison at this size is measured "
            "against whichever side is wrong (KI-BACKEND-006)."
            % (spread, n, values["cpu"], values["cuda"]))


if __name__ == "__main__":
    unittest.main()
