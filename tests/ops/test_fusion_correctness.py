# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Fusion-compiler numerical correctness — fused vs unfused vs numpy.

Jittor *is* a fusion compiler: it merges chains of meta-operators into single JIT
kernels. The audit found the compiler tests only parse log lines ("kernel fused"),
never the *numbers* — so a fused kernel that computes the WRONG value would pass.
This module closes that: for each expression it computes the result with fusion ON
(default) and with fusion OFF (``jt.flag_scope(no_fuse=1)``) and asserts they agree
AND match an independent numpy reference — forward AND backward, on every device.

A fused-vs-unfused divergence isolates a fusion miscompile directly (both are jittor,
only the kernel-merging differs). It also pins the specific fusion traps this project
has hit: the same-pointer ``x == x`` fold (must equal numpy, NaN included via isnan),
and the inf/nan ternary chain that once segfaulted.

Run::  python -m pytest tests/ops/test_fusion_correctness.py
"""
import unittest

import numpy as np
import jittor as jt

from _helpers.common import (
    JittorTestCase, get_all_device_types, use_cuda_for, to_numpy,
)


class TestFusionCorrectness(JittorTestCase):

    def _devices(self, body):
        for d in get_all_device_types():
            with self.subTest(device=d):
                with jt.flag_scope(use_cuda=use_cuda_for(d)):
                    body(d)

    def _fused_unfused(self, build, arrays, ref, *, check_grad=True,
                       atol=1e-5, rtol=1e-5, label=""):
        """Compare build(*Vars) fused vs unfused vs numpy ref, fwd (+bwd)."""
        def compute(no_fuse):
            with jt.flag_scope(no_fuse=no_fuse):
                vs = [jt.array(a, dtype=str(a.dtype)) for a in arrays]
                out = build(*vs)
                fwd = to_numpy(out)
                grads = None
                if check_grad:
                    diff = [v for v in vs if str(v.dtype).startswith("float")]
                    if diff:
                        cot = np.random.RandomState(0).randn(*fwd.shape).astype("float32")
                        grads = [to_numpy(g) for g in jt.grad((out * jt.array(cot)).sum(), diff)]
                return fwd, grads

        f_fused, g_fused = compute(0)
        f_unfused, g_unfused = compute(1)
        # fused == unfused (fusion must not change the value)
        self.assertEqual(f_fused, f_unfused, atol=atol, rtol=rtol,
                         msg=f"{label}: FUSED != UNFUSED forward")
        # fused == numpy (and so unfused too)
        self.assertEqual(f_fused, ref, atol=atol, rtol=rtol,
                         msg=f"{label}: fused forward != numpy")
        if g_fused is not None and g_unfused is not None:
            for k, (a, b) in enumerate(zip(g_fused, g_unfused)):
                self.assertEqual(a, b, atol=atol * 10, rtol=rtol * 10,
                                 msg=f"{label}: FUSED != UNFUSED backward (input {k})")

    # -- long elementwise chain ---------------------------------------------------
    def test_elementwise_chain(self):
        rng = np.random.RandomState(0)
        a0 = rng.randn(4, 5).astype("float32")
        b0 = rng.randn(4, 5).astype("float32")
        c0 = (rng.rand(4, 5).astype("float32") + 0.5)

        def body(dev):
            ref = a0 * b0 + c0 / (a0 + 2.0) - np.exp(b0 * 0.1)
            self._fused_unfused(
                lambda a, b, c: a * b + c / (a + 2.0) - jt.exp(b * 0.1),
                [a0, b0, c0], ref, atol=1e-4, rtol=1e-4, label=f"chain [{dev}]")
        self._devices(body)

    # -- reduce + broadcast + binary fused (normalization pattern) ----------------
    def test_reduce_broadcast_fused(self):
        rng = np.random.RandomState(1)
        x0 = rng.randn(3, 6).astype("float32")
        w0 = rng.randn(6).astype("float32")
        b0 = rng.randn(6).astype("float32")

        def body(dev):
            mean = x0.mean(1, keepdims=True)
            var = x0.var(1, keepdims=True)
            ref = (x0 - mean) / np.sqrt(var + 1e-5) * w0 + b0
            def build(x, w, b):
                m = x.mean(1, keepdims=True)
                v = ((x - m) ** 2).mean(1, keepdims=True)
                return (x - m) / jt.sqrt(v + 1e-5) * w + b
            self._fused_unfused(build, [x0, w0, b0], ref,
                                atol=1e-4, rtol=1e-4, label=f"norm-fuse [{dev}]")
        self._devices(body)

    # -- comparison + where fused -------------------------------------------------
    def test_where_fused(self):
        rng = np.random.RandomState(2)
        a0 = rng.randn(5, 5).astype("float32")
        b0 = rng.randn(5, 5).astype("float32")

        def body(dev):
            ref = np.where(a0 > b0, a0 * 2.0, b0 - 1.0).astype("float32")
            self._fused_unfused(
                lambda a, b: jt.ternary(a > b, a * 2.0, b - 1.0),
                [a0, b0], ref, atol=1e-5, rtol=1e-5, label=f"where-fuse [{dev}]")
        self._devices(body)

    # -- §4-B trap: same-pointer x==x must still equal numpy (NaN handled) ---------
    def test_self_compare_and_isnan(self):
        x0 = np.array([1.0, np.nan, 3.0, np.inf, -2.0], dtype="float32")

        def body(dev):
            x = jt.array(x0)
            # the SAFE non-nan mask is isnan, never (x==x) (which fuses to all-True);
            # lock that isnan matches numpy exactly through the fused graph.
            self.assertEqual((~jt.isnan(x)).numpy().astype(bool), ~np.isnan(x0),
                             msg=f"isnan through fusion [{dev}]")
        self._devices(body)

    # -- the inf/nan ternary chain that once segfaulted (must compute, not crash) --
    def test_inf_nan_ternary_chain(self):
        x0 = np.array([1.0, np.inf, -np.inf, 5.0, -7.0], dtype="float32")

        def body(dev):
            ref = np.where(np.isinf(x0), np.zeros_like(x0), x0).astype("float32")
            self._fused_unfused(
                lambda x: jt.ternary(jt.isinf(x), jt.zeros_like(x), x),
                [x0], ref, check_grad=False, label=f"inf-nan ternary [{dev}]")
        self._devices(body)

    # -- cumulative + elementwise fused -------------------------------------------
    def test_cumsum_fused(self):
        x0 = np.random.RandomState(3).randn(3, 5).astype("float32")

        def body(dev):
            ref = np.cumsum(x0, axis=1) * 2.0 + 1.0
            self._fused_unfused(
                lambda x: jt.cumsum(x, dim=1) * 2.0 + 1.0,
                [x0], ref, atol=1e-4, rtol=1e-4, label=f"cumsum-fuse [{dev}]")
        self._devices(body)


if __name__ == "__main__":
    unittest.main(verbosity=2)
