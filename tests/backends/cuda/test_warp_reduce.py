# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Reductions that end in an atomic add reduce inside the warp first.

The pass rewrites ``atomicAdd(&yp[yid], acc)`` into a warp shuffle reduction
followed by one atomic per warp. It is only allowed to do that when the whole
warp is present and every lane agrees on the index, and the generated code
checks both at run time, so these cases cover the shapes that take the fast
path and the shapes that fall back.
"""

from _helpers.runtime_policy import preserve_policy as _test_preserve_policy

from _helpers import capability as _test_capability
import unittest

import numpy as np

import jittor as jt


@_test_preserve_policy(jt, 'use_cuda')
@unittest.skipIf(not _test_capability.check_accelerator('cuda', backend=jt).enabled, "No cuda found")
class TestWarpReduce(unittest.TestCase):
    def setUp(self):
        from contextlib import ExitStack as _TestPolicyStack
        _test_policy_stack = _TestPolicyStack()
        self.addCleanup(_test_policy_stack.close)
        self._saved = jt.introspection.policy.runtime.use_cuda
        _test_policy_stack.enter_context(jt.runtime.scope(use_cuda=1))

    def tearDown(self):
        from contextlib import ExitStack as _TestPolicyStack
        with _TestPolicyStack() as _test_policy_stack:
            _test_policy_stack.enter_context(jt.runtime.scope(use_cuda=self._saved))

    def _check(self, shape, dims, dtype="float32", tol=1e-5):
        data = np.random.RandomState(abs(hash((shape, tuple(dims)))) % 2**31)
        value = data.randn(*shape).astype(dtype)
        # jt.array(value) alone narrows float64 to float32; name the dtype.
        got = jt.array(value, dtype=dtype).sum(dims).numpy()
        expected = value.sum(axis=tuple(dims))
        scale = max(1.0, float(np.abs(expected).max()))
        error = float(np.abs(got.reshape(expected.shape) - expected).max()) / scale
        self.assertLess(error, tol, "%s over %s" % (shape, dims))

    def test_trailing_spatial_dims(self):
        # The diffusers UNet backward shape the pass was written for.
        self._check((8, 96, 32, 32), [2, 3])

    def test_large_reduction_per_output(self):
        self._check((4, 32, 64, 64), [2, 3])

    def test_small_reduction_per_output(self):
        # Few elements per output: the warp may not be whole, so this exercises
        # the fallback branch.
        self._check((8, 64, 2, 2), [2, 3])

    def test_leading_dims(self):
        self._check((16, 8, 4, 4), [0, 2, 3])

    def test_full_reduction(self):
        self._check((64, 128), [0, 1])

    def test_single_dim(self):
        self._check((129, 37), [0])

    def test_float64(self):
        self._check((4, 16, 8, 8), [2, 3], dtype="float64", tol=1e-12)

    def test_gradient_through_reduction(self):
        value = np.random.RandomState(3).randn(4, 8, 16, 16).astype("float32")
        x = jt.array(value)
        x.start_grad()
        loss = (x * 2).sum([2, 3]).sum()
        grad = jt.grad(loss, x).numpy()
        np.testing.assert_allclose(grad, np.full_like(value, 2.0), rtol=1e-6)

    def test_repeated_runs_are_stable(self):
        # The same reduction must not drift between launches.
        value = np.random.RandomState(5).randn(8, 64, 16, 16).astype("float32")
        first = jt.array(value).sum([2, 3]).numpy()
        for _ in range(3):
            again = jt.array(value).sum([2, 3]).numpy()
            np.testing.assert_allclose(again, first, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
