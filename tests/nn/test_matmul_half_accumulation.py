# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# Maintainers: Dun Liang <randonlang@gmail.com>.
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Half-precision matmul accumulates in float32, independent of AMP bits.

The generic contraction is ``(a * b).sum(-1)`` under a reduce scope. When the
scope carried both ``keep_reduce`` and a half preference, the reduce
constructor's explicit float32 cast came back as a half Var, so the product
accumulated at the half width: at K=256 the error against a float64 reference
built from the same half inputs grew to 3.3% for bfloat16, and an autocast
training run drifted far beyond torch's bf16 spread. ``keep_reduce`` alone
keeps the public result at the input dtype while the accumulator stays
float32.
"""

import unittest

import numpy as np

import jittor as jt


def _reference(a, b):
    return (a.float32().numpy().astype(np.float64)
            @ b.float32().numpy().astype(np.float64))


class TestHalfMatmulAccumulation(unittest.TestCase):
    def _check(self, dtype, tolerance):
        rng = np.random.RandomState(7)
        a = jt.array(rng.randn(64, 256).astype("float32") * .1).cast(dtype)
        b = jt.array(rng.randn(256, 32).astype("float32") * .1).cast(dtype)
        reference = _reference(a, b)
        for amp in (0, jt.amp_flags.keep_reduce,
                    jt.amp_flags.prefer16 | jt.amp_flags.keep_reduce,
                    jt.amp_flags.prefer_bfloat16 | jt.amp_flags.keep_reduce):
            with self.subTest(amp=amp):
                with jt.flag_scope(amp_reg=amp):
                    result = a @ b
                self.assertEqual(str(result.dtype),
                                 "bfloat16" if dtype == "bfloat16" else "float16")
                got = result.float32().numpy().astype(np.float64)
                relative = np.max(np.abs(got - reference)) / np.max(np.abs(reference))
                self.assertLess(
                    relative, tolerance,
                    "%s matmul under amp_reg=%d accumulated at the half width: "
                    "%.3e" % (dtype, amp, relative))

    def test_bfloat16(self):
        self._check("bfloat16", 5e-3)

    def test_float16(self):
        self._check("float16", 5e-3)


if __name__ == "__main__":
    unittest.main()
