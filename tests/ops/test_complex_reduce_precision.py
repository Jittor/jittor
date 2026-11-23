# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""A complex64 sum is as accurate as the float32 sum of its parts.

The CUDA float sum folds each warp with shuffles before its one atomic; the
complex sum used to skip that and send every lane's partial to the output with
two float atomics, so the norm of a 2^20-amplitude state was 1e-5 off where the
float32 sum of the same probabilities was 4e-8 off. A rank-20 view of the same
state was additionally summed by 16 threads, each walking 2^16 terms.
"""

from _helpers.runtime_policy import preserve_policy as _test_preserve_policy
from _helpers import capability as _test_capability
import unittest

import numpy as np

import jittor as jt


def _state(n=20, seed=0):
    rs = np.random.RandomState(seed)
    z = rs.randn(2 ** n) + 1j * rs.randn(2 ** n)
    return (z / np.linalg.norm(z)).astype(np.complex64)


@unittest.skipIf(not _test_capability.check_accelerator("cuda", backend=jt).enabled,
                 "no usable CUDA in this build")
@_test_preserve_policy(jt, "use_cuda")
class TestComplexReducePrecisionCuda(unittest.TestCase):
    """Sums on CUDA; the gradient of a complex full sum is checked too."""
    def setUp(self):
        jt.flags.use_cuda = 1

    def _rel(self, got, z):
        z = z.astype(np.complex128)
        ref = np.sum(z * z.conj())
        return abs(complex(got) - ref) / abs(ref)

    def test_flat_complex_sum(self):
        z = _state()
        x = jt.array(z)
        # float32 accumulation of 2^20 terms; the float32 sum of the same
        # magnitudes lands at ~1.2e-7 on this data.
        self.assertLess(self._rel((x * x.conj()).sum().numpy(), z), 2.5e-7)
        self.assertLess(self._rel(jt.sum(x * x.conj()).numpy(), z), 2.5e-7)

    def test_rank20_complex_sum(self):
        z = _state()
        x = jt.array(z.reshape([2] * 20))
        self.assertLess(self._rel((x * x.conj()).sum().numpy(), z), 2.5e-7)
        self.assertLess(self._rel((x.conj() * x).real.sum().numpy(), z), 2.5e-7)

    def test_complex_full_sum_gradient(self):
        rs = np.random.RandomState(3)
        a = (rs.randn(1 << 15) + 1j * rs.randn(1 << 15)).astype(np.complex64)
        x = jt.array(a)
        g = jt.grad((x.sum() * jt.array(np.complex64(2 - 1j))).real, x).numpy()
        np.testing.assert_allclose(g, np.full_like(a, 2 + 1j), rtol=1e-6)

    def test_partial_complex_sum_matches_numpy(self):
        rs = np.random.RandomState(1)
        a = (rs.randn(64, 4096) + 1j * rs.randn(64, 4096)).astype(np.complex64)
        got = jt.array(a).sum(1).numpy()
        ref = a.astype(np.complex128).sum(1)
        np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
