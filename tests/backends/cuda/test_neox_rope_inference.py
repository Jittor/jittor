# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""CUDA inference fast path for NeoX-style rotary position embeddings."""

from _helpers import capability as _test_capability
import unittest

import numpy as np

import jittor as jt
from jittor._runtime.dispatch import try_dispatch


def _reference_neox_rope(x, cos, sin, dtype):
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    rh = np.concatenate((-x2, x1), axis=-1)
    if dtype == "float32":
        return (x * cos + rh * sin).astype(np.float32)
    # Match half/bf16 rounding of (x * cos) + (rotate_half(x) * sin)
    jx = jt.array(x).cast(dtype)
    jcos = jt.array(cos).cast(dtype)
    jsin = jt.array(sin).cast(dtype)
    jrh = jt.concat((-jx[..., half:], jx[..., :half]), dim=-1)
    return (jx * jcos + jrh * jsin).float32().numpy()


@unittest.skipIf(
    not _test_capability.check_accelerator("cuda", backend=jt).enabled,
    "no usable CUDA in this build",
)
class TestNeoxRopeInference(unittest.TestCase):
    def setUp(self):
        self.scope = jt.flag_scope(use_cuda=1)
        self.scope.__enter__()
        self.addCleanup(self.scope.__exit__, None, None, None)
        self.rs = np.random.RandomState(0)

    def _check(self, shape, cos_shape, dtype, seq_major=False):
        b, h, s, d = shape
        if seq_major:
            base = self.rs.randn(b, s, h, d).astype(np.float32)
            jbase = jt.array(base).cast(dtype)
            jx = jbase.transpose(0, 2, 1, 3)
            xr = np.transpose(jbase.float32().numpy(), (0, 2, 1, 3))
        else:
            x = self.rs.randn(b, h, s, d).astype(np.float32)
            jx = jt.array(x).cast(dtype)
            xr = jx.float32().numpy()
        cos = self.rs.randn(*cos_shape).astype(np.float32)
        sin = self.rs.randn(*cos_shape).astype(np.float32)
        jcos = jt.array(cos).cast(dtype)
        jsin = jt.array(sin).cast(dtype)
        cr = jcos.float32().numpy()
        sr = jsin.float32().numpy()
        with jt.no_grad():
            got = try_dispatch("nn.neox_rope.inference", jx, jcos, jsin)
        self.assertIsNotNone(got, f"dispatch declined {shape} {dtype} seq_major={seq_major}")
        got.sync()
        self.assertEqual(got.location(), "device")
        want = _reference_neox_rope(xr, cr, sr, dtype)
        tol = 1e-5 if dtype == "float32" else 1e-3
        np.testing.assert_allclose(
            got.float32().numpy(),
            want,
            rtol=tol,
            atol=tol,
            err_msg=f"shape={shape} cos={cos_shape} dtype={dtype} seq_major={seq_major}",
        )

    def test_vectorized_and_scalar_shapes(self):
        for dtype in ("float32", "float16", "bfloat16"):
            for d in (64, 96, 128, 132):
                for seq_major in (False, True):
                    self._check((2, 8, 33, d), (1, 1, 33, d), dtype, seq_major=seq_major)

    def test_batched_cos_sin(self):
        for dtype in ("float32", "bfloat16"):
            self._check((2, 4, 17, 64), (2, 1, 17, 64), dtype, seq_major=True)

    def test_declines_with_grad(self):
        x = jt.randn((1, 4, 8, 64), dtype="float32").start_grad()
        c = jt.randn((1, 1, 8, 64), dtype="float32")
        s = jt.randn((1, 1, 8, 64), dtype="float32")
        self.assertIsNone(try_dispatch("nn.neox_rope.inference", x, c, s))


    def test_pending_broadcast_factors(self):
        shape = (2, 3, 5, 16)
        values = self.rs.randint(-4, 5, size=shape).astype(np.float32)
        cosine = self.rs.randint(-4, 5, size=(1, 5, 16)).astype(np.float32)
        sine = self.rs.randint(-4, 5, size=(1, 5, 16)).astype(np.float32)
        with jt.flag_scope(backend_fallback="error", transpose_storage_view=1,
                           auto_flush_ops=0, async_flush_ops=0), jt.no_grad():
            x = jt.array(values.transpose(0, 2, 1, 3).copy()).bfloat16().transpose(0, 2, 1, 3)
            cos = jt.array(cosine).bfloat16().broadcast(shape).clone()
            sin = jt.array(sine).bfloat16().broadcast(shape).clone()
            out = try_dispatch("nn.neox_rope.inference", x, cos, sin)
            self.assertIsNotNone(out)
            out.sync()
            self.assertEqual(out.location(), "device")
            self.assertFalse(out.requires_grad)
            rotated = np.concatenate((-values[..., 8:], values[..., :8]), axis=-1)
            np.testing.assert_array_equal(out.float32().numpy(), values * cosine + rotated * sine)


if __name__ == "__main__":
    unittest.main()
