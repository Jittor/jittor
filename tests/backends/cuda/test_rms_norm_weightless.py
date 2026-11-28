# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""CUDA weightless RMSNorm inference unit tests."""

from _helpers import capability as _test_capability
import unittest

import numpy as np

import jittor as jt
from jittor._runtime.dispatch import try_dispatch


def _reference_rms_norm(x, eps):
    x64 = x.astype(np.float64)
    return x64 / np.sqrt(np.mean(x64 * x64, axis=-1, keepdims=True) + eps)


@unittest.skipIf(
    not _test_capability.check_accelerator("cuda", backend=jt).enabled,
    "no usable CUDA in this build",
)
class TestWeightlessRmsNormInference(unittest.TestCase):
    def setUp(self):
        self.scope = jt.flag_scope(use_cuda=1)
        self.scope.__enter__()
        self.addCleanup(self.scope.__exit__, None, None, None)
        self.rs = np.random.RandomState(0)

    def test_subwarp_block_and_scalar_widths(self):
        for dtype in ("bfloat16", "float16", "float32"):
            for hidden in (64, 128, 256, 300, 1024, 1536, 4096):
                x = (self.rs.randn(37, hidden) * 2.0 + 0.3).astype(np.float32)
                jx = jt.array(x).cast(dtype)
                xr = jx.float64().numpy()
                with jt.no_grad():
                    got = try_dispatch("nn.rms_norm.weightless.inference", jx, 1e-6)
                self.assertIsNotNone(got, f"declined {dtype} hidden={hidden}")
                got.sync()
                self.assertEqual(got.location(), "device")
                want = _reference_rms_norm(xr, 1e-6)
                tol = 1e-5 if dtype == "float32" else 2e-2
                np.testing.assert_allclose(
                    got.float64().numpy(),
                    want,
                    rtol=tol,
                    atol=tol * max(1.0, np.abs(want).max()),
                    err_msg=f"{dtype} hidden={hidden}",
                )


if __name__ == "__main__":
    unittest.main()
