# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""The CUDA ``fused_adamw`` operator, against the AdamW formula in numpy.

One operator updates a whole parameter list, where the per-parameter update
built about ten graph nodes per tensor: 4500 for a diffusers UNet every step,
which made its AdamW step cost more host time than the forward. The list below
is longer than one launch's 36-tensor table and has sizes on and off the
per-block width, including an empty tensor, so the table's chunking and each
tensor's tail are both exercised.
"""

from _helpers import capability as _test_capability
import unittest

import numpy as np

import jittor as jt


def _adamw(p, m, v, g, step, lr, b1, b2, wd, eps):
    p = p * (1 - lr * wd)
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * g * g
    p = p - m * (lr / (1 - b1 ** step)) / (np.sqrt(v) / np.sqrt(1 - b2 ** step) + eps)
    return p, m, v


@unittest.skipIf(not _test_capability.check_accelerator("cuda", backend=jt).enabled,
                 "no usable CUDA in this build")
class TestFusedAdamwCuda(unittest.TestCase):
    def test_a_list_longer_than_one_launch(self):
        rng = np.random.RandomState(0)
        sizes = [1, 7, 1024, 1025, 0, 4096 * 3 + 5] * 7          # 42 tensors
        hyper = dict(lr=1e-2, b1=0.9, b2=0.999, wd=0.1, eps=1e-8)
        arrays = [[rng.randn(n).astype("float32") for n in sizes] for _ in range(4)]
        # Variances away from zero: m / sqrt(v) over a tiny v amplifies float32
        # rounding far beyond what real optimizer state produces.
        arrays[2] = [np.abs(a) + 0.1 for a in arrays[2]]
        with jt.flag_scope(use_cuda=1):
            p, m, v, g = ([jt.array(a) for a in family] for family in arrays)
            step = jt.array(3.0).stop_grad()
            out = jt.fused_adamw(p, m, v, g, step, hyper["lr"], hyper["b1"], hyper["b2"],
                                 hyper["wd"], hyper["eps"])
            got = [o.numpy() for o in out]
        count = len(sizes)
        for i in range(count):
            want = _adamw(*(family[i].astype(np.float64) for family in arrays), 3.0, **hyper)
            for name, value, expected in zip("pmv", (got[i], got[count + i], got[2 * count + i]), want):
                # float32 arithmetic against a float64 reference: a few ulps.
                np.testing.assert_allclose(value, expected, rtol=2e-6, atol=5e-7,
                                           err_msg="%s[%d]" % (name, i))


if __name__ == "__main__":
    unittest.main()
