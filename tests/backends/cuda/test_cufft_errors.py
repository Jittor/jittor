# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""A failed cuFFT call must raise, not print and carry on.

``CUFFT_CALL`` used to write the failure to stderr and continue.  A failed
``cufftPlanMany`` therefore left an invalid handle behind, the op stored that
handle in its plan cache and executed with it, and the caller got undefined
output with no exception -- and every later transform of the same shape reused
the same invalid handle.
"""
import unittest

import numpy as np

import jittor as jt
from jittor import nn


def _reference_fft2(real):
    spectrum = np.fft.fft2(real, axes=(1, 2))
    return np.stack([spectrum.real, spectrum.imag], axis=-1)


@unittest.skipIf(not jt.has_cuda, "No CUDA found")
class TestCufftErrors(unittest.TestCase):
    def setUp(self):
        self._use_cuda = jt.flags.use_cuda
        jt.flags.use_cuda = 1

    def tearDown(self):
        jt.sync_all()
        jt.flags.use_cuda = self._use_cuda

    def test_forward_matches_numpy(self):
        real = np.random.RandomState(0).rand(2, 8, 8).astype("float32")
        x = jt.array(np.stack([real, np.zeros_like(real)], axis=-1))
        got = nn._fft2(x).numpy()
        np.testing.assert_allclose(got, _reference_fft2(real), atol=1e-3,
                                   rtol=1e-3)

    def test_invalid_plan_raises(self):
        # A zero-length transform dimension makes cufftPlanMany fail with
        # CUFFT_INVALID_SIZE. Nothing usable can come out of that, so the op
        # has to report it.
        with self.assertRaises(RuntimeError):
            nn._fft2(jt.zeros((1, 0, 4, 2), "float32")).sync()

    def test_invalid_plan_is_not_cached(self):
        # The first failure must not leave an invalid handle in the plan
        # cache: a second attempt with the same shape has to fail the same
        # way rather than "succeed" through the cached garbage.
        for _ in range(2):
            with self.assertRaises(RuntimeError):
                nn._fft2(jt.zeros((1, 4, 0, 2), "float32")).sync()

    def test_valid_transforms_still_work_after_a_failure(self):
        with self.assertRaises(RuntimeError):
            nn._fft2(jt.zeros((1, 0, 4, 2), "float32")).sync()
        real = np.random.RandomState(3).rand(1, 4, 4).astype("float32")
        x = jt.array(np.stack([real, np.zeros_like(real)], axis=-1))
        np.testing.assert_allclose(nn._fft2(x).numpy(), _reference_fft2(real),
                                   atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    unittest.main()
