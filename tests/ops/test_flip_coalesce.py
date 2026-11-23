# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""flip reindexes the coalesced shape: adjacent dims flipped alike are one.

A [2]*20 state flipped on one axis used to be a rank-20 reindex, 370x slower
than the same bytes seen as rank 3. These check the coalesced form gives the
same values and gradients on every grouping of flipped, unflipped and unit dims.
"""

from _helpers.runtime_policy import preserve_policy as _test_preserve_policy
from _helpers import capability as _test_capability
import unittest

import numpy as np

import jittor as jt


_CASES = [
    ([2] * 20, [0]),
    ([2] * 20, [10]),
    ([2] * 20, [19]),
    ([2] * 20, [3, 4, 5]),
    ([2] * 20, [1, 7, 18]),
    ([2] * 20, list(range(20))),
    ([3, 1, 4, 1, 5], [1]),
    ([3, 1, 4, 1, 5], [0, 1, 2]),
    ([3, 1, 4, 1, 5], [3, 4]),
    ([1, 1], [0]),
    ([6], [0]),
]


@_test_preserve_policy(jt, "use_cuda")
class TestFlipCoalesce(unittest.TestCase):
    use_cuda = 0

    def setUp(self):
        jt.flags.use_cuda = self.use_cuda

    def test_values_and_grads_match_numpy(self):
        rs = np.random.RandomState(0)
        for shape, dims in _CASES:
            for dtype in ("float32", "complex64"):
                with self.subTest(shape=shape, dims=dims, dtype=dtype):
                    a = rs.randn(*shape).astype(np.float32)
                    if dtype == "complex64":
                        a = (a + 1j * rs.randn(*shape)).astype(np.complex64)
                    w = rs.randn(*shape).astype(np.float32)
                    x = jt.array(a)
                    y = x.flip(dims)
                    np.testing.assert_array_equal(y.numpy(), np.flip(a, dims))
                    loss = (y.real * jt.array(w)).sum() if dtype == "complex64" \
                        else (y * jt.array(w)).sum()
                    g = jt.grad(loss, x).numpy()
                    np.testing.assert_allclose(g.real if dtype == "complex64" else g,
                                               np.flip(w, dims), rtol=1e-6)

    def test_flip_of_a_strided_view(self):
        a = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
        x = jt.array(a).permute(2, 0, 1)
        np.testing.assert_array_equal(x.flip([1, 2]).numpy(),
                                      np.flip(a.transpose(2, 0, 1), [1, 2]))


@unittest.skipIf(not _test_capability.check_accelerator("cuda", backend=jt).enabled,
                 "no usable CUDA in this build")
class TestFlipCoalesceCuda(TestFlipCoalesce):
    use_cuda = 1


if __name__ == "__main__":
    unittest.main()
