# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""The CUDA layer norm that serves calls recording no gradient.

Two kernels answer, by shape: a warp per row for many rows of at most 1024
values, a block per row otherwise. Both keep a double-precision pass for a
row whose float sums overflow. Each is checked here against float64 numpy,
with an affine given as tensors and as plain numbers, on widths off the warp
size, and with one row large enough to take the double pass.
"""

from _helpers import capability as _test_capability
import unittest

import numpy as np

import jittor as jt
from jittor.backends.cuda.kernels.nn import layer_norm_cuda as _ln


def _reference(x, w, b, eps):
    x = x.astype(np.float64)
    mean = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * w + b


@unittest.skipIf(not _test_capability.check_accelerator("cuda", backend=jt).enabled,
                 "no usable CUDA in this build")
class TestLayerNormInference(unittest.TestCase):
    def setUp(self):
        self.scope = jt.flag_scope(use_cuda=1)
        self.scope.__enter__()
        self.addCleanup(self.scope.__exit__, None, None, None)
        self.rs = np.random.RandomState(0)

    def _check(self, rows, hidden, dtype, tensor_affine, huge_row=False):
        x = (self.rs.randn(rows, hidden) * 3 + 1).astype("float32")
        if huge_row:
            # Float sums of this row overflow; its values do not.
            x[rows // 2] = (self.rs.randn(hidden) * 1e37).astype("float32")
        w = self.rs.randn(hidden).astype("float32")
        b = self.rs.randn(hidden).astype("float32")
        jx = jt.array(x).cast(dtype)
        xr = jx.float64().numpy()
        if tensor_affine:
            jw, jb = jt.array(w).cast(dtype), jt.array(b).cast(dtype)
            wr, br = jw.float64().numpy(), jb.float64().numpy()
        else:
            jw, jb, wr, br = 1.5, -0.25, 1.5, -0.25
        with jt.no_grad():
            got = _ln._layer_norm_no_grad_cuda(jx, (hidden,), jw, jb, 1e-5)
        want = _reference(xr, wr, br, 1e-5)
        tol = 1e-5 if dtype == "float32" else 2e-2
        np.testing.assert_allclose(got.float64().numpy(), want, rtol=tol,
                                   atol=tol * max(1.0, np.abs(want).max()),
                                   err_msg=f"{rows}x{hidden} {dtype} tensor_affine={tensor_affine}")

    def test_the_warp_per_row_kernel(self):
        for hidden in (320, 300, 1000):
            self.assertTrue(_ln._warp_rows(jt.zeros((2048, hidden)), hidden))
            for dtype in ("float32", "float16"):
                for tensor_affine in (True, False):
                    self._check(2048, hidden, dtype, tensor_affine)

    def test_the_block_per_row_kernel(self):
        for rows, hidden in ((128, 768), (1024, 2048)):
            self.assertFalse(_ln._warp_rows(jt.zeros((rows, hidden)), hidden))
            for tensor_affine in (True, False):
                self._check(rows, hidden, "float32", tensor_affine)

    def test_a_row_whose_float_sums_overflow(self):
        for rows, hidden in ((2048, 320), (128, 768)):
            for tensor_affine in (True, False):
                self._check(rows, hidden, "float32", tensor_affine, huge_row=True)


if __name__ == "__main__":
    unittest.main()
