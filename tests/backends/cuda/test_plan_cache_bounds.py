# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""cuFFT and cuTT plan caches must stay bounded.

Both caches used to keep one plan per distinct shape forever, and each plan
owns device memory, so a workload that keeps meeting new shapes never stopped
growing.  The cuFFT op leaked a second plan on top of that: ``cufftCreate``
produced a handle that the following ``cufftPlanMany`` immediately overwrote
and nothing ever destroyed.
"""
import unittest

import numpy as np

import jittor as jt
from jittor.nn.legacy_complex import _fft2


cufft = getattr(jt, "cufft", None)
cutt = getattr(jt.compile_extern, "cutt", None)

_HAS_CUFFT = cufft is not None and hasattr(cufft, "cufft_plan_cache_size")
_HAS_CUTT = cutt is not None and hasattr(cutt, "cutt_plan_cache_size")


def _complex_input(batch, n1, n2):
    real = np.random.RandomState(n1 * 1000 + n2).rand(batch, n1, n2)
    return jt.array(np.stack([real, np.zeros_like(real)], axis=-1).astype("float32"))


def _reference_fft2(x):
    real = x.numpy()[..., 0]
    spectrum = np.fft.fft2(real, axes=(1, 2))
    return np.stack([spectrum.real, spectrum.imag], axis=-1)


@unittest.skipIf(not jt.has_cuda, "No CUDA found")
@unittest.skipIf(not _HAS_CUFFT, "cuFFT plan cache accessors not built")
class TestCufftPlanCacheBounds(unittest.TestCase):
    def setUp(self):
        self._use_cuda = jt.flags.use_cuda
        jt.flags.use_cuda = 1

    def tearDown(self):
        jt.sync_all()
        cufft.cufft_set_plan_cache_size(32)
        jt.flags.use_cuda = self._use_cuda

    def test_same_shape_reuses_one_plan(self):
        cufft.cufft_set_plan_cache_size(8)
        before = cufft.cufft_plan_cache_size()
        for _ in range(4):
            _fft2(_complex_input(2, 12, 12)).sync()
        self.assertEqual(cufft.cufft_plan_cache_size(), before + 1)

    def test_many_shapes_stay_within_the_limit(self):
        cufft.cufft_set_plan_cache_size(4)
        for n in range(6, 24):
            x = _complex_input(1, n, n)
            got = _fft2(x).numpy()
            np.testing.assert_allclose(got, _reference_fft2(x), atol=1e-2,
                                       rtol=1e-2)
            self.assertLessEqual(cufft.cufft_plan_cache_size(), 4)

    def test_evicted_shape_is_rebuilt_and_still_correct(self):
        cufft.cufft_set_plan_cache_size(1)
        first = _complex_input(1, 8, 8)
        _fft2(first).sync()
        _fft2(_complex_input(1, 16, 16)).sync()   # evicts the 8x8 plan
        np.testing.assert_allclose(_fft2(first).numpy(),
                                   _reference_fft2(first), atol=1e-2, rtol=1e-2)
        self.assertEqual(cufft.cufft_plan_cache_size(), 1)


@unittest.skipIf(not jt.has_cuda, "No CUDA found")
@unittest.skipIf(not _HAS_CUTT, "cuTT plan cache accessors not built")
class TestCuttPlanCacheBounds(unittest.TestCase):
    def setUp(self):
        self._use_cuda = jt.flags.use_cuda
        jt.flags.use_cuda = 1

    def tearDown(self):
        jt.sync_all()
        cutt.cutt_set_plan_cache_size(64)
        jt.flags.use_cuda = self._use_cuda

    def test_many_shapes_stay_within_the_limit(self):
        cutt.cutt_set_plan_cache_size(4)
        for n in range(3, 20):
            a = np.arange(n * (n + 1) * 2).reshape(n, n + 1, 2).astype("float32")
            got = jt.transpose(jt.array(a), (2, 0, 1)).numpy()
            np.testing.assert_allclose(got, np.transpose(a, (2, 0, 1)))
            self.assertLessEqual(cutt.cutt_plan_cache_size(), 4)

    def test_evicted_shape_is_rebuilt_and_still_correct(self):
        cutt.cutt_set_plan_cache_size(1)
        a = np.arange(4 * 5 * 6).reshape(4, 5, 6).astype("float32")
        b = np.arange(7 * 8 * 9).reshape(7, 8, 9).astype("float32")
        jt.transpose(jt.array(a), (2, 0, 1)).sync()
        jt.transpose(jt.array(b), (2, 0, 1)).sync()   # evicts the first plan
        np.testing.assert_allclose(
            jt.transpose(jt.array(a), (2, 0, 1)).numpy(),
            np.transpose(a, (2, 0, 1)))
        self.assertEqual(cutt.cutt_plan_cache_size(), 1)


if __name__ == "__main__":
    unittest.main()
