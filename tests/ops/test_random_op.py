
from _helpers import capability as _test_capability
# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved. 
# Maintainers: 
#     Guoye Yang <498731903@qq.com>
#     Dun Liang <randonlang@gmail.com>. 
# 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import jittor as jt
from jittor import nn, Module
from jittor.models import vgg, resnet
import numpy as np
import sys, os
import random
import math
import unittest
from _helpers.logs import find_log_with_re
from _helpers.assertions import expect_error
from _helpers.torch_runtime import import_torch_modules, modules_available
from _helpers.tuner_parser import simple_parser

skip_this_test = not modules_available("torch")
torch = None


def setUpModule():
    global torch
    if not skip_this_test:
        (torch,) = import_torch_modules("torch")


class TestRandomOp(unittest.TestCase):
    def test_invalid_type_is_a_catchable_user_error(self):
        expect_error(
            lambda: jt.random((2,), type="invalid"),
            exc_type=RuntimeError,
            match="ns_uniform",
        )

    @unittest.skipIf(not _test_capability.check_accelerator('cuda', backend=jt).enabled, "Cuda not found")
    @jt.flag_scope(use_cuda=1)
    def test(self):
        jt.set_seed(3)
        with jt.log_capture_scope(
            log_silent=1,
            log_v=0, log_vprefix="op.cc=100"
        ) as raw_log:
            t = jt.random([5,5])
            t.data
        logs = find_log_with_re(raw_log, "(Jit op key (not )?found: " + "curand_random" + ".*)")
        assert len(logs)==1

    @unittest.skipIf(not _test_capability.check_accelerator('cuda', backend=jt).enabled, "Cuda not found")
    @jt.flag_scope(use_cuda=1)
    def test_float64(self):
        jt.set_seed(3)
        with jt.log_capture_scope(
            log_silent=1,
            log_v=0, log_vprefix="op.cc=100"
        ) as raw_log:
            t = jt.random([5,5], dtype='float64')
            t.data
        logs = find_log_with_re(raw_log, "(Jit op key (not )?found: " + "curand_random" + ".*)")
        assert len(logs)==1

    @unittest.skipIf(skip_this_test, "No Torch Found")
    def test_normal(self):
        from jittor import init
        n = 10000
        r = 0.155
        a = init.gauss([n], "float32", 1, 3)
        data = a.data

        assert (np.abs((data<(1-3)).mean() - r) < 0.1)
        assert (np.abs((data<(1)).mean() - 0.5) < 0.1)
        assert (np.abs((data<(1+3)).mean() - (1-r)) < 0.1)

        np_res = np.random.normal(1, 0.1, (100, 100))
        jt_res = jt.normal(1., 0.1, (100, 100))
        assert (np.abs(np_res.mean() - jt_res.data.mean()) < 0.1)
        assert (np.abs(np_res.std() - jt_res.data.std()) < 0.1)

        np_res = torch.normal(torch.arange(1., 10000.), 1)
        jt_res = jt.normal(jt.arange(1, 10000), 1)
        assert (np.abs(np_res.mean() - jt_res.data.mean()) < 0.1)
        assert (np.abs(np_res.std() - jt_res.data.std()) < 1)

        np_res = np.random.randn(100, 100)
        jt_res = jt.randn(100, 100)
        assert (np.abs(np_res.mean() - jt_res.data.mean()) < 0.1)
        assert (np.abs(np_res.std() - jt_res.data.std()) < 0.1)

        np_res = np.random.rand(100, 100)
        jt_res = jt.rand(100, 100)
        assert (np.abs(np_res.mean() - jt_res.data.mean()) < 0.1)
        assert (np.abs(np_res.std() - jt_res.data.std()) < 0.1)

    @unittest.skipIf(not _test_capability.check_accelerator('cuda', backend=jt).enabled, "Cuda not found")
    @jt.flag_scope(use_cuda=1)
    def test_normal_cuda(self):
        self.test_normal()

    def test_other_rand(self):
        a = jt.array([1.0,2.0,3.0])
        b = jt.rand_like(a)
        c = jt.randn_like(a)
        assert b.shape == c.shape
        assert b.shape == a.shape
        print(b, c)
        assert jt.randint(10, 20, (2000,)).min() == 10
        assert jt.randint(10, 20, (2000,)).max() == 19
        assert jt.randint(10, shape=(2000,)).max() == 9
        assert jt.randint_like(a, 10).shape == a.shape

    def _check_seed_is_reproducible(self):
        jt.set_seed(731)
        first = jt.random((8,)).numpy()
        jt.set_seed(731)
        second = jt.random((8,)).numpy()
        np.testing.assert_array_equal(
            first, second, err_msg="set_seed did not reproduce the sequence")
        jt.set_seed(99)
        other = jt.random((8,)).numpy()
        assert not np.array_equal(first, other), \
            "a different seed produced the same sequence"

    def test_seed_is_reproducible(self):
        self._check_seed_is_reproducible()

    @unittest.skipIf(not _test_capability.check_accelerator('cuda', backend=jt).enabled, "Cuda not found")
    @jt.flag_scope(use_cuda=1)
    def test_seed_is_reproducible_cuda(self):
        # curand keeps its position in the sequence across a re-seed, so
        # setting the same seed again used to continue from wherever the
        # previous draw left off instead of starting over.
        self._check_seed_is_reproducible()


    def _check_linspace_endpoint(self):
        """The last value is ``end`` exactly, at every length and placement.

        numpy and torch both promise this, and callers compare against it. The
        arithmetic series alone leaves the final point one rounding step off.
        """
        for start in (0.0, 1.0, -1.0, 2.5):
            for end in (0.0, 1.0, -1.0, 2.5):
                for steps in (2, 3, 4, 5, 6, 8, 17, 50):
                    with self.subTest(start=start, end=end, steps=steps):
                        got = jt.linspace(start, end, steps).numpy()
                        self.assertEqual(tuple(got.shape), (steps,))
                        # exact in the result's own dtype, not allclose
                        self.assertEqual(got[-1], got.dtype.type(end))
                        self.assertEqual(got[0], got.dtype.type(start))
                        # The interior is float32 arithmetic accumulated over
                        # ``steps`` terms, against a schedule numpy evaluates in
                        # float64: a point that lands mathematically on zero can
                        # miss it by ~1e-7 (seen at index 35 of
                        # linspace(2.5, -1.0, 50)). Only the endpoint is a
                        # contract; the rest is tolerance.
                        np.testing.assert_allclose(
                            got, np.linspace(start, end, steps),
                            rtol=1e-5, atol=1e-5)

    def test_linspace_endpoint_is_exact(self):
        self._check_linspace_endpoint()

    @unittest.skipIf(not _test_capability.check_accelerator('cuda', backend=jt).enabled, "Cuda not found")
    @jt.flag_scope(use_cuda=1)
    def test_linspace_endpoint_is_exact_cuda(self):
        self._check_linspace_endpoint()

    def test_a_descending_linspace_does_not_undershoot_its_end(self):
        """A schedule that ends below zero breaks its consumers.

        MiniMax-H3 builds its sigma schedule from ``linspace(1.0, 0.0, n)`` and
        rejects a negative ``sigma_next``, so a last point of ``-2.98e-08``
        (which is what the CPU placement produced for n=4) failed every request
        with 4 or more denoise steps. The CUDA placement rounded that point to
        exactly 0, which is why the bug only showed on the CPU one.
        """
        for steps in (4, 5, 6, 8, 17, 50):
            with self.subTest(steps=steps):
                last = float(jt.linspace(1.0, 0.0, steps).numpy()[-1])
                self.assertGreaterEqual(last, 0.0)
                self.assertEqual(last, 0.0)

    def test_linspace_single_step_is_the_start(self):
        for start in (0.0, 3.5):
            with self.subTest(start=start):
                got = jt.linspace(start, 9.0, 1).numpy()
                self.assertEqual(tuple(got.shape), (1,))
                self.assertEqual(got[0], got.dtype.type(start))


if __name__ == "__main__":
    unittest.main()
