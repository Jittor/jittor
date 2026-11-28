# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Vectorized CUDA inference fast path for modulated LayerNorm."""

from _helpers import capability as _test_capability
import unittest

import numpy as np

import jittor as jt
from jittor._runtime.dispatch import try_dispatch


def _reference_modulated_ln(x, scale, shift, eps, one_plus):
    x64 = x.astype(np.float64)
    mean = x64.mean(-1, keepdims=True)
    var = x64.var(-1, keepdims=True)
    norm = (x64 - mean) / np.sqrt(var + eps)
    s = (1.0 + scale.astype(np.float64)) if one_plus else scale.astype(np.float64)
    return norm * s + shift.astype(np.float64)


@unittest.skipIf(
    not _test_capability.check_accelerator("cuda", backend=jt).enabled,
    "no usable CUDA in this build",
)
class TestModulatedLayerNormInference(unittest.TestCase):
    def setUp(self):
        self.scope = jt.flag_scope(use_cuda=1)
        self.scope.__enter__()
        self.addCleanup(self.scope.__exit__, None, None, None)
        self.rs = np.random.RandomState(0)

    def _check(self, x_shape, mod_shape, dtype, one_plus=True, huge_row=False):
        x = (self.rs.randn(*x_shape) * 2.0 + 0.5).astype(np.float32)
        if huge_row:
            x[0, 0] = (self.rs.randn(x_shape[-1]) * 1e37).astype(np.float32)
        sc = (self.rs.randn(*mod_shape) * 0.25).astype(np.float32)
        sh = (self.rs.randn(*mod_shape) * 0.25).astype(np.float32)
        jx = jt.array(x).cast(dtype)
        jsc = jt.array(sc).cast(dtype)
        jsh = jt.array(sh).cast(dtype)
        xr, scr, shr = jx.float64().numpy(), jsc.float64().numpy(), jsh.float64().numpy()
        with jt.no_grad():
            got = try_dispatch("nn.layer_norm.modulated", jx, jsc, jsh, 1e-6, one_plus=one_plus)
        self.assertIsNotNone(got, f"declined {x_shape} {mod_shape} {dtype} one_plus={one_plus}")
        got.sync()
        self.assertEqual(got.location(), "device")
        want = _reference_modulated_ln(xr, scr, shr, 1e-6, one_plus)
        tol = 1e-5 if dtype == "float32" else 3e-2
        np.testing.assert_allclose(
            got.float64().numpy(),
            want,
            rtol=tol,
            atol=tol * max(1.0, np.abs(want).max()),
            err_msg=f"{x_shape} {mod_shape} {dtype} one_plus={one_plus}",
        )

    def test_vectorized_and_scalar_shapes(self):
        for dtype in ("bfloat16", "float16", "float32"):
            for hidden in (190, 768, 1536, 3072):
                self._check((2, 17, hidden), (2, 1, hidden), dtype, one_plus=True)
            self._check((1, 32, 1536), (1, 1, 1536), dtype, one_plus=False)
            self._check((16, 1536), (1536,), dtype, one_plus=True)

    def test_overflow_row_double_fallback(self):
        self._check((2, 8, 768), (2, 1, 768), "float32", one_plus=True, huge_row=True)


if __name__ == "__main__":
    unittest.main()
