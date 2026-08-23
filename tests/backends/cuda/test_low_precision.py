# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Low-precision (fp16 / bf16) gradient-dtype preservation.

The audit named this a highest-bug-value gap: a backward that silently upcasts to
float32 and forgets to cast the gradient back to fp16/bf16 *passes every forward
check* — forward dtype is correct, only the gradient dtype is wrong, which breaks
mixed-precision training (the optimizer expects `param.grad.dtype == param.dtype`).
The legacy `test_fp16`/`test_bf16` are forward/partial and do not lock this.

These tests run on the accelerator (fp16/bf16 are GPU/NPU dtypes) and assert the
gradient dtype matches the input dtype, the way torch's autograd does.

Run::  python -m pytest tests/backends/cuda/test_low_precision.py
"""
import unittest

import numpy as np
import jittor as jt

from _helpers.common import JittorTestCase, HAS_CUDA, HAS_ACL

F = jt.nn.functional


def _supports(dtype):
    """Whether this build can construct/cast to ``dtype`` on the accelerator."""
    try:
        with jt.flag_scope(use_cuda=1):
            v = jt.array(np.zeros(2, "float32")).cast(dtype)
            v.sync()
        return True
    except Exception:
        return False


@unittest.skipUnless(HAS_CUDA, "low-precision dtypes need an accelerator (CUDA/NPU)")
class TestLowPrecisionGradDtype(JittorTestCase):

    def _grad_dtype(self, dtype, build):
        with jt.flag_scope(use_cuda=1):
            x = jt.array(np.random.RandomState(0).randn(4, 8).astype("float32")).cast(dtype)
            out = build(x)
            g = jt.grad(out.sum(), [x])[0]
            return str(out.dtype), str(g.dtype)

    def test_matmul_grad_dtype_preserved(self):
        # matmul backward keeps the low-precision dtype (verified correct). Lock it.
        w = None
        for dtype in ("float16", "bfloat16"):
            if not _supports(dtype):
                continue
            with jt.flag_scope(use_cuda=1):
                wmat = jt.array(np.random.RandomState(1).randn(8, 4).astype("float32")).cast(dtype)
            od, gd = self._grad_dtype(dtype, lambda x, wmat=wmat: jt.matmul(x, wmat))
            self.assertEqual(od, dtype, msg=f"{dtype} matmul out dtype")
            self.assertEqual(gd, dtype,
                             msg=f"{dtype} matmul grad dtype must match input (got {gd})")

    def test_elementwise_grad_dtype(self):
        # Elementwise backward preserves the input dtype, including scalar multiply
        # where the upstream reduction gradient is commonly float32.
        for dtype in ("float16", "bfloat16"):
            if not _supports(dtype):
                self.skipTest(f"{dtype} unsupported")
            for name, build in (("relu", lambda x: jt.nn.relu(x)),
                                ("relu_mul", lambda x: jt.nn.relu(x) * 2),
                                ("mul", lambda x: x * 2)):
                od, gd = self._grad_dtype(dtype, build)
                self.assertEqual(od, dtype, msg=f"{dtype} {name} output dtype")
                self.assertEqual(gd, dtype,
                                 msg=f"{dtype} {name} grad dtype should match input (got {gd})")


if __name__ == "__main__":
    unittest.main(verbosity=2)
