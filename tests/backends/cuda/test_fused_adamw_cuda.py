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


@unittest.skipIf(not _test_capability.check_accelerator("cuda", backend=jt).enabled,
                 "no usable CUDA in this build")
class TestFusedAdamwScalarPrecision(unittest.TestCase):
    def test_zero_state_complements_and_parameter_update_match_independent_cuda(self):
        from jittor._runtime import forbid_backend_fallbacks

        # Independent PyTorch 2.8 CUDA foreach AdamW, float32 state. These
        # gradients distinguish float(1-beta) from 1-float(beta), and the
        # parameter fixture binds the divide/addcdiv rounding order too.
        gradient = np.array([.0033893585205078125, -.004253387451171875,
                             .002529144287109375, .00069427490234375,
                             -.00551605224609375, -.00047016143798828125,
                             .0006160736083984375, -.0014486312866210938], dtype="float32")
        expected_parameter = np.array([-9.999969915952533e-5, 9.999977191910148e-5,
                                       -9.999960457207635e-5, -9.999856411013752e-5,
                                       9.999982285080478e-5, 9.999786561820656e-5,
                                       -9.999836765928194e-5, 9.999931353377178e-5], dtype="float32")
        expected_moment = np.float32(1.0 - .9) * gradient
        expected_variance = np.float32(1.0 - .999) * (gradient * gradient)
        with jt.flag_scope(use_cuda=1, amp_reg=0), forbid_backend_fallbacks():
            parameter = jt.zeros((8,))
            moment, variance = jt.zeros((8,)), jt.zeros((8,))
            outputs = jt.fused_adamw([parameter], [moment], [variance], [jt.array(gradient)],
                                    jt.array(1.).stop_grad(), .0001, .9, .999, .01, 1e-8)
            for value, expected in zip(outputs, [expected_parameter, expected_moment, expected_variance]):
                value.sync()
                self.assertEqual(value.location(), "device")
                np.testing.assert_array_equal(value.numpy(), expected)
