# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""What each backend does with a subnormal, asserted rather than assumed.

float32's smallest normal is ``1.18e-38``. Smaller non-zero values -- the
subnormals -- are kept by CPU, as IEEE-754 says, and **flushed to zero** by
CUDA, because nvcc's ``--use_fast_math`` implies ``-ftz=true``
(KI-BACKEND-008).

That is a decision, not a mistake: subnormal arithmetic is slow and most
training does not care. What made it expensive is that it was written down
nowhere, so every device-parity comparison that hit it read as a defect, and
the two backends were documented as equivalent when they are not. The
consequence reaches past the value itself -- the input is already zero when it
arrives at the function, so ``log(1e-45)`` is ``-103.28`` on CPU and ``-inf``
on CUDA, and ``count_nonzero`` disagrees on the same array.

This file is not an argument that either behaviour is right. It states what the
behaviour *is*, so that a change to it is visible. That includes a change for
the better: if the default ever becomes ``strict``, this test goes red, and the
thing to fix is this file and ``docs/notes/float32-precision-policy.md``, not
to let them keep describing a world that no longer holds.

The ``strict`` rows are the load-bearing ones. Without them the file would
assert only that CUDA flushes, which is satisfied by a build where the policy
switch does nothing at all.
"""

from _helpers import capability as _test_capability

import unittest

import numpy as np

import jittor as jt


def _has_cuda():
    return bool(_test_capability.check_accelerator('cuda', backend=jt).enabled)


#: Two subnormals and one normal, so a build that flushed *everything* small
#: could not pass by flushing the subnormals it was asked about.
SUBNORMALS = (1e-45, 1e-40)
SMALLEST_NORMAL = 1.18e-38
NORMAL = 1e-30


def _survives(value, use_cuda, policy=None):
    """Is ``value`` still non-zero after one arithmetic op on this device?

    Multiplied by one rather than read back directly: the question is what the
    *kernel* does with it, and a plain copy would not go through one.
    """
    scope = {"use_cuda": use_cuda}
    if policy is not None:
        scope["cuda_kernel_math"] = policy
    with jt.flag_scope(**scope):
        x = jt.array(np.full(8, value, dtype="float32"))
        # KI-DTYPE-002: jt.array narrows 64-bit values; say which width this is.
        assert str(x.dtype) == "float32", x.dtype
        return float((x * 1.0).numpy()[0])


class TestSubnormalContractCpu(unittest.TestCase):

    def test_cpu_keeps_subnormals(self):
        for value in SUBNORMALS:
            with self.subTest(value=value):
                self.assertNotEqual(
                    _survives(value, 0), 0.0,
                    "CPU flushed %g to zero; IEEE-754 and NumPy keep it" % value)

    def test_the_test_values_really_are_subnormal(self):
        """Otherwise this file would be asserting nothing about subnormals."""
        for value in SUBNORMALS:
            self.assertLess(value, SMALLEST_NORMAL)
        self.assertGreater(NORMAL, SMALLEST_NORMAL)
        self.assertNotEqual(np.float32(NORMAL), np.float32(0.0))


@unittest.skipIf(not _has_cuda(), "no CUDA device")
class TestSubnormalContractCuda(unittest.TestCase):

    def test_cuda_flushes_subnormals_by_default(self):
        for value in SUBNORMALS:
            with self.subTest(value=value):
                self.assertEqual(
                    _survives(value, 1, "default"), 0.0,
                    "CUDA kept %g. That is better than the documented "
                    "behaviour, not worse -- update this test and "
                    "docs/notes/float32-precision-policy.md rather than "
                    "reverting it." % value)

    def test_cuda_keeps_normals(self):
        """The flush is limited to subnormals, not to everything small."""
        self.assertNotEqual(_survives(NORMAL, 1, "default"), 0.0)

    def test_strict_policy_restores_subnormals(self):
        """The escape hatch has to actually work, or the row above is a dead end."""
        for value in SUBNORMALS:
            with self.subTest(value=value):
                self.assertNotEqual(
                    _survives(value, 1, "strict"), 0.0,
                    "cuda_kernel_math='strict' did not stop the flush for %g; "
                    "the policy switch is not reaching the kernel flags"
                    % value)

    def test_the_flush_reaches_functions_not_just_values(self):
        """The expensive half: the input is already zero when the op sees it.

        A test that only checked the stored value would pass on a build that
        flushed at the boundary and computed correctly inside, which is not
        what happens and not what callers experience.
        """
        with jt.flag_scope(use_cuda=1, cuda_kernel_math="default"):
            flushed = float(jt.log(jt.array(
                np.full(8, 1e-45, dtype="float32"))).numpy()[0])
        with jt.flag_scope(use_cuda=1, cuda_kernel_math="strict"):
            kept = float(jt.log(jt.array(
                np.full(8, 1e-45, dtype="float32"))).numpy()[0])
        self.assertEqual(flushed, float("-inf"),
                         "log of a flushed subnormal should be -inf")
        self.assertTrue(np.isfinite(kept) and kept < -100,
                        "log(1e-45) under strict should be about -103.28, got %s"
                        % kept)


if __name__ == "__main__":
    unittest.main()
