# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Encoded edge-case batteries for jittor's known kernel traps (CONTEXT §4-B).

These are not "ordinary op" tests -- they pin down the *inherent behaviors and
gotchas* that have repeatedly produced silent-wrong results, so that any future
change which trips one of them fails loudly. Each test documents the trap and the
correct usage. Sources are the project's kernel-trap ledger and the fixes that
addressed them (isfinite/isnan composition ``f769e288``, dtype-promotion lattice
``f1c400c6``, negative-dim normalization ``eaec3b9c``/``3eb7bc78``).

Run::  python -m pytest tests/compiler/test_kernel_traps.py
"""
import unittest

import numpy as np
import jittor as jt

from _helpers.common import (
    JittorTestCase, get_all_device_types, use_cuda_for,
)


class TestKernelTraps(JittorTestCase):

    def _devices(self, body):
        for d in get_all_device_types():
            with self.subTest(device=d):
                with jt.flag_scope(use_cuda=use_cuda_for(d)):
                    body(d)

    # -- §4-B#2: x==x (same Var) folds to all-True; NaN masking must use isnan ----
    def test_nan_handling_isfinite_isnan_isinf(self):
        # f769e288: isnan/isinf/isfinite must be IEEE-correct (they are composed to
        # avoid the x==x fusion fold that would mis-classify NaN as finite).
        x = np.array([1.0, np.inf, -np.inf, np.nan, 0.0, -3.5], dtype="float32")

        def body(dev):
            v = jt.array(x)
            self.assertEqual(jt.isnan(v).numpy().astype(bool), np.isnan(x),
                             msg=f"isnan [{dev}]")
            self.assertEqual(jt.isinf(v).numpy().astype(bool), np.isinf(x),
                             msg=f"isinf [{dev}]")
            self.assertEqual(jt.isfinite(v).numpy().astype(bool), np.isfinite(x),
                             msg=f"isfinite [{dev}]")
        self._devices(body)

    def test_nan_not_equal_to_itself_via_isnan(self):
        # The safe NaN mask is jt.isnan(x), NOT (x == x): the latter is fused to
        # all-True for a self-compare (CONTEXT §4-B#2). Lock that isnan disagrees
        # with the naive self-compare exactly on the NaN element.
        x = np.array([1.0, np.nan, 3.0], dtype="float32")

        def body(dev):
            v = jt.array(x)
            non_nan_mask = (~jt.isnan(v)).numpy().astype(bool)
            self.assertEqual(non_nan_mask, ~np.isnan(x), msg=f"isnan mask [{dev}]")
        self._devices(body)

    # -- §4-B#3 / 3eb7bc78 / eaec3b9c: negative dims must normalize, not crash -----
    def test_negative_dim_normalized(self):
        x = np.random.RandomState(0).randn(2, 3, 4).astype("float32")

        def body(dev):
            v = jt.array(x)
            am = jt.argmax(v, dim=-1)
            am = am[0] if isinstance(am, tuple) else am   # torch-compat returns the Var
            self.assertEqual(am.numpy(), x.argmax(axis=-1),
                             msg=f"argmax dim=-1 [{dev}]")
            self.assertEqual(jt.flip(v, [-1]).numpy(), np.flip(x, -1),
                             msg=f"flip dim=-1 [{dev}]")
            self.assertEqual(jt.cumsum(v, -1).numpy(), np.cumsum(x, -1),
                             atol=1e-5, rtol=1e-5, msg=f"cumsum dim=-1 [{dev}]")
        self._devices(body)

    def test_index_select_negative_dim(self):
        x = np.random.RandomState(0).randn(2, 3, 4).astype("float32")
        indices = jt.array(np.array([0, 2], "int64"))
        for dim in (-1, -2, 2):
            with self.subTest(dim=dim):
                got = jt.index_select(jt.array(x), dim, indices).numpy()
                self.assertEqual(
                    got,
                    np.take(x, [0, 2], axis=dim),
                    msg=f"index_select dim={dim}",
                )

        for dim in (-4, 3):
            with self.subTest(dim=dim):
                with self.assertRaises(IndexError):
                    jt.index_select(jt.array(x), dim, indices)

    # -- §4-B#1: jittor has no 0-d scalar -- a full reduce is shape (1,) not () ----
    def test_no_zero_d_scalar(self):
        x = jt.array(np.random.RandomState(1).randn(3, 4).astype("float32"))
        # This is an inherent jittor convention; lock it so code relying on it (and
        # the references that np.atleast_1d to match it) stay consistent.
        self.assertEqual(tuple(jt.sum(x).shape), (1,), msg="full-reduce sum is (1,)")
        self.assertEqual(tuple(x.mean().shape), (1,), msg="full-reduce mean is (1,)")

    # -- the float64 narrowing trap (found while building this suite) --------------
    def test_jt_array_float64_narrowing(self):
        # jt.array(float64_ndarray) silently narrows to float32 (torch_compat default).
        # This is a real trap for any double-precision reference (e.g. gradcheck): the
        # dtype MUST be pinned explicitly. Lock both the behavior and the workaround so
        # a future change to either is noticed.
        a = np.zeros(3, dtype="float64")
        self.assertEqual(str(jt.array(a).dtype), "float32",
                         msg="jt.array(f64) narrows to float32 (known trap)")
        self.assertEqual(str(jt.array(a, dtype="float64").dtype), "float64",
                         msg="jt.array(a, dtype='float64') preserves double (workaround)")

    # -- #10 / f1c400c6: torch dtype-promotion lattice -----------------------------
    def test_dtype_promotion_lattice(self):
        # NB: jt.array narrows int64->int32 (and float64->float32), so dtypes MUST be
        # pinned explicitly to actually exercise the int64 lattice.
        i32 = jt.array(np.array([1, 2, 3], dtype="int32"), dtype="int32")
        i64 = jt.array(np.array([1, 2, 3], dtype="int64"), dtype="int64")
        f32 = jt.array(np.array([1.0, 2.0, 3.0], dtype="float32"), dtype="float32")
        # int32 + int64 -> int64 (wider int wins)
        self.assertEqual(str((i32 + i64).dtype), "int64", msg="int32+int64 -> int64")
        # float32 + int64 -> float32 (float beats int regardless of width)
        self.assertEqual(str((f32 + i64).dtype), "float32", msg="float32+int64 -> float32")
        # true division of ints -> float (never integer division)
        self.assertEqual(str((i32 / i32).dtype)[:5], "float", msg="int/int -> float")

    # -- #10: .long() must be int64 (was mis-aliased to int32) ---------------------
    def test_long_is_int64(self):
        x = jt.array(np.array([1, 2, 3], dtype="int32"))
        if hasattr(x, "long"):
            self.assertEqual(str(x.long().dtype), "int64", msg=".long() -> int64")
        else:
            self.skipTest(".long() not exposed")

    @unittest.expectedFailure
    def test_constant_pad_fractional_fill_cpu_asmtuner(self):
        """KNOWN-BUG (expected failure): a constant-pad with a FRACTIONAL fill value
        fails to COMPILE on CPU.

        Found by the full op battery: ``nn.pad(..., mode='constant', value=0.7)`` on the
        CPU backend emits a ``reindex`` kernel whose overflow value is the hex-float
        ``itof(0x3fe6666666666666)``; jittor's CPU ``asm_tuner`` (the ``-march=native``
        assembly optimizer) rewrites that constant into a malformed literal and g++ aborts
        with ``error: exponent has no digits``. INTEGER fills (0.0, 2.0, -3.0) and the CUDA
        backend are unaffected -- so op_db's pad_constant samples use integer fills to keep
        the pad semantics covered, and this test pins the fractional-fill regression loudly.
        When the asm_tuner is fixed this turns XPASS and we drop the xfail."""
        with jt.flag_scope(use_cuda=0):
            x = jt.array(np.random.RandomState(0).randn(1, 1, 4, 4).astype("float32"))
            out = jt.nn.pad(x, [1, 1, 1, 1], mode="constant", value=0.7)
            out.sync()        # force compile+exec; raises on the asm_tuner failure


if __name__ == "__main__":
    unittest.main(verbosity=2)
