# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Gradients of many slices of one var accumulate in place.

Each slice's gradient used to be a zero-filled buffer the size of the sliced
var, a setitem into it, and a full-size add onto the running sum: 400 scalar
slices of a [10, 2, 20] parameter took 3,186 kernels. The running sum now takes
each slice's gradient by one setitem-add. These check the values on every kind
of slice and mix, and the kernel count on CUDA.
"""

from _helpers.runtime_policy import preserve_policy as _test_preserve_policy
from _helpers import capability as _test_capability
import unittest

import numpy as np

import jittor as jt
from _helpers.common import kernel_launches


@_test_preserve_policy(jt, "use_cuda")
class TestGetitemGradAccumulate(unittest.TestCase):
    use_cuda = 0

    def setUp(self):
        jt.flags.use_cuda = self.use_cuda

    def test_scalar_slices(self):
        a = np.random.RandomState(0).randn(10, 2, 20).astype("float32")
        p = jt.array(a)
        loss = 0
        for i in range(10):
            for j in range(2):
                for k in range(0, 20, 3):
                    loss = loss + p[i, j, k] * p[i, j, k]
        want = np.zeros_like(a)
        want[:, :, ::3] = 2 * a[:, :, ::3]
        np.testing.assert_allclose(jt.grad(loss, p).numpy(), want, rtol=1e-6)

    def test_mixed_slices_and_other_uses(self):
        rs = np.random.RandomState(1)
        a = rs.randn(6, 5).astype("float32")
        w = rs.randn(6, 5).astype("float32")
        p = jt.array(a)
        idx = jt.array(np.array([0, 2, 2, 5], "int32"))
        loss = (p * jt.array(w)).sum()          # a whole-tensor gradient first
        loss = loss + p[1].sum() * 3            # a row
        loss = loss + (p[:, 1:4] ** 2).sum()    # a range
        loss = loss + p[idx].sum()              # an index var, with a repeat
        loss = loss + p[4, 3] * 7               # a scalar
        want = w.copy()
        want[1] += 3
        want[:, 1:4] += 2 * a[:, 1:4]
        for r in (0, 2, 2, 5):
            want[r] += 1
        want[4, 3] += 7
        np.testing.assert_allclose(jt.grad(loss, p).numpy(), want, rtol=1e-5)

    def test_slices_first_then_whole(self):
        a = np.arange(12, dtype="float32").reshape(3, 4)
        p = jt.array(a)
        loss = p[0, 0] * 2 + p[2, 3] * 5 + p.sum()
        want = np.ones_like(a)
        want[0, 0] += 2
        want[2, 3] += 5
        np.testing.assert_allclose(jt.grad(loss, p).numpy(), want)

    def test_many_integer_slices_are_batched(self):
        # Enough one-element slices to take the stacked scatter: negative and
        # repeated indices, mixed with a whole-tensor and a range contribution.
        rs = np.random.RandomState(4)
        a = rs.randn(3, 4, 5).astype("float32")
        p = jt.array(a)
        picks = [(0, 1, 2), (-1, -1, -1), (0, 1, 2), (2, 0, 4), (1, 3, 0), (1, -2, 3)]
        loss = (p * 0.5).sum() + (p[1:, :2] ** 2).sum()
        want = np.full_like(a, 0.5)
        want[1:, :2] += 2 * a[1:, :2]
        for i, (x, y, z) in enumerate(picks):
            loss = loss + p[x, y, z] * (i + 1)
            want[x, y, z] += i + 1
        np.testing.assert_allclose(jt.grad(loss, p).numpy(), want, rtol=1e-5)

    def test_batched_slices_have_a_second_derivative(self):
        a = np.arange(1, 9, dtype="float32")
        p = jt.array(a)
        loss = 0
        for k in range(8):
            loss = loss + p[k] ** 3
        g = jt.grad(loss, p)
        np.testing.assert_allclose(g.numpy(), 3 * a ** 2, rtol=1e-5)
        np.testing.assert_allclose(jt.grad(g.sum(), p).numpy(), 6 * a, rtol=1e-5)

    def test_complex_slices(self):
        rs = np.random.RandomState(2)
        a = (rs.randn(4, 3) + 1j * rs.randn(4, 3)).astype("complex64")
        p = jt.array(a)
        loss = 0
        for i in range(4):
            loss = loss + (p[i, 1] * p[i, 1].conj()).real
        g = jt.grad(loss, p).numpy()
        want = np.zeros_like(a)
        want[:, 1] = 2 * a[:, 1]
        np.testing.assert_allclose(g, want, rtol=1e-5, atol=1e-6)


@unittest.skipIf(not _test_capability.check_accelerator("cuda", backend=jt).enabled,
                 "no usable CUDA in this build")
class TestGetitemGradAccumulateCuda(TestGetitemGradAccumulate):
    use_cuda = 1

    def test_scalar_slices_do_not_build_full_size_buffers(self):
        p = jt.full((10, 2, 20), 0.1, dtype="float32")
        p.sync()

        def step():
            loss = 0
            for i in range(10):
                for j in range(2):
                    for k in range(20):
                        loss = loss + p[i, j, k] * p[i, j, k]
            return jt.grad(loss, p)
        count, g = kernel_launches(lambda: step().numpy())
        np.testing.assert_allclose(g, np.full((10, 2, 20), 0.2, "float32"), rtol=1e-6)
        # 800 slice reads: their gradients are stacked and scattered once, and
        # the per-slice products share kernels (3,186 kernels before).
        self.assertLess(count, 300)


if __name__ == "__main__":
    unittest.main()
