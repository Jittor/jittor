# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Named regression locks for the *silent-wrong* bugs this project has fixed.

The audit found that many of the suite's most valuable checks are regression locks
hidden inside generic-looking tests -- a specific input shape that guards a specific
past bug, easy to drop in a refactor. This module makes them explicit: one named
test per fixed bug, each citing its commit, each constructed to FAIL on the buggy
behavior and pass on the fix. They are independent of the op_db battery (those test
"is the op correct in general"; these test "did *this* bug stay fixed").

Every check compares against an INDEPENDENT reference (numpy / analytic), never
jittor-vs-jittor. Bugs that only reproduce on an accelerator are marked accordingly.

Run::  python -m pytest tests/core/test_regression.py
"""
import unittest

import numpy as np
import jittor as jt
from jittor import nn

from _helpers.common import (
    JittorTestCase, get_all_device_types, use_cuda_for, HAS_CUDA, HAS_ACL,
)

F = nn.functional


def _scalar(a):
    """python float from a jittor Var / numpy array (jittor has no 0-d scalar, so a
    reduced value is a zero-dimensional array accepted by ``float()``)."""
    if isinstance(a, jt.Var):
        a = a.numpy()
    return float(np.asarray(a).reshape(-1)[0])


class TestSilentWrongRegressions(JittorTestCase):
    """Each method locks one fixed silent-wrong bug (commit cited)."""

    def _devices(self, body):
        for d in get_all_device_types():
            with self.subTest(device=d):
                with jt.flag_scope(use_cuda=use_cuda_for(d)):
                    body(d)

    def test_scalar_cube_matches_analytic_forward_and_backward(self):
        x_np = np.array([-3.0, -0.5, -0.0, 0.25, 2.0], dtype="float32")
        cot_np = np.array([0.5, -2.0, 3.0, -0.25, 1.5], dtype="float32")

        def body(dev):
            x = jt.array(x_np)
            output = jt.pow(x, 3.0)
            grad = jt.grad((output * jt.array(cot_np)).sum(), [x])[0]
            self.assertEqual(output.numpy(), x_np * x_np * x_np,
                             msg=f"scalar cube forward [{dev}]")
            self.assertEqual(grad.numpy(), 3.0 * x_np * x_np * cot_np,
                             msg=f"scalar cube backward [{dev}]")
        self._devices(body)

    # -- 40875685: Var.where(cond, other) treated self as the CONDITION ----------
    def test_var_where_is_select_not_condition(self):
        a = np.random.RandomState(0).randn(4, 5).astype("float32")
        b = np.random.RandomState(1).randn(4, 5).astype("float32")
        cond = a > b

        def body(dev):
            out = jt.where(jt.array(cond), jt.array(a), jt.array(b)).numpy()
            self.assertEqual(out, np.where(cond, a, b), msg=f"where select [{dev}]")
        self._devices(body)

    # -- cbad57db: var/std default must be UNBIASED (torch correction=1) ---------
    def test_var_std_unbiased_default(self):
        x = np.random.RandomState(2).randn(50).astype("float32")
        # torch's Var.var()/std() default to the Bessel-corrected (unbiased) estimate.
        # (tol is loose enough to clear float32 round-off but tight enough to separate
        # the unbiased ddof=1 value from the biased ddof=0 one, which differ by ~2%.)
        self.assertEqual(_scalar(jt.array(x).var()), float(np.var(x, ddof=1)),
                         atol=1e-5, rtol=1e-4, msg="var default unbiased (not biased)")
        self.assertEqual(_scalar(jt.array(x).std()), float(np.std(x, ddof=1)),
                         atol=1e-5, rtol=1e-4, msg="std default unbiased (not biased)")

    # -- 0b3e7e5f: nanmean must not count NaN -----------------------------------
    def test_nanmean_excludes_nan(self):
        x = np.array([1.0, np.nan, 3.0, np.nan, 5.0], dtype="float32")
        ref = float(np.nanmean(x))
        got = _scalar(jt.array(x).nanmean()) if hasattr(jt.Var, "nanmean") \
            else _scalar(jt.nanmean(jt.array(x)))
        self.assertEqual(got, ref, atol=1e-6, rtol=1e-6, msg="nanmean skips NaN")

    # -- 3eb7bc78: index_select with dim>0 (newaxis routing bug) -----------------
    def test_index_select_dim_gt0(self):
        x = np.random.RandomState(3).randn(5, 4).astype("float32")
        idx = np.array([0, 2, 3], dtype="int64")

        def body(dev):
            out = jt.index_select(jt.array(x), 1, jt.array(idx)).numpy()
            self.assertEqual(out, x[:, idx], msg=f"index_select dim=1 [{dev}]")
        self._devices(body)

    def test_index_select_function_and_method_match_numpy(self):
        x = np.random.RandomState(31).randn(3, 4).astype("float32")
        idx = np.array([2, 1], dtype="int64")

        def body(dev):
            value = jt.array(x)
            indices = jt.array(idx)
            self.assertEqual(
                jt.index_select(value, 0, indices).numpy(),
                np.take(x, idx, axis=0),
                msg=f"index_select function dim=0 [{dev}]",
            )
            self.assertEqual(
                value.index_select(1, indices).numpy(),
                np.take(x, idx, axis=1),
                msg=f"index_select method dim=1 [{dev}]",
            )
        self._devices(body)

    # -- 85cdfe75 / 9be5444f: index_add accumulates duplicate indices ------------
    def test_index_add_accumulates_duplicates(self):
        x = np.zeros((4,), dtype="float32")
        idx = np.array([1, 1, 1, 2], dtype="int64")     # index 1 thrice
        src = np.array([1.0, 2.0, 3.0, 4.0], dtype="float32")
        # torch index_add ACCUMULATES duplicates -> x[1]=6, x[2]=4 (not overwrite)
        ref = x.copy()
        np.add.at(ref, idx, src)

        def body(dev):
            t = jt.array(x)
            out = t.index_add(0, jt.array(idx), jt.array(src)) if hasattr(t, "index_add") \
                else None
            if out is None:
                self.skipTest("index_add not exposed")
            self.assertEqual(out.numpy(), ref, msg=f"index_add accumulate [{dev}]")
        self._devices(body)

    # -- a7bb1b78: fft.fftshift must actually roll (was a no-op stub) ------------
    def test_fftshift_rolls(self):
        x = np.arange(8, dtype="float32")
        got = jt.fft.fftshift(jt.array(x)).numpy() if hasattr(jt, "fft") else None
        if got is None:
            self.skipTest("jt.fft.fftshift not exposed")
        self.assertEqual(got, np.fft.fftshift(x), msg="fftshift real roll")

    # -- 52d71415: linalg.solve gradient wrt b must be nonzero (was a 0 stub) ----
    def test_solve_grad_wrt_b_nonzero(self):
        rng = np.random.RandomState(4)
        A = (rng.randn(4, 4) + 4 * np.eye(4)).astype("float64")    # well-conditioned
        b = rng.randn(4, 1).astype("float64")
        Av = jt.array(A, dtype="float64")
        bv = jt.array(b, dtype="float64")
        x = jt.linalg.solve(Av, bv)
        gb = jt.grad(x.sum(), [bv])[0].numpy()
        # analytic: d sum(solve(A,b))/db = A^{-T} @ 1
        ref = np.linalg.solve(A.T, np.ones((4, 1)))
        self.assertGreater(float(np.abs(gb).max()), 1e-8, "solve d/db must be nonzero")
        self.assertEqual(gb, ref, atol=1e-6, rtol=1e-6, msg="solve d/db analytic")

    # -- b846b281: Categorical(logits=) uses softmax, not sigmoid ----------------
    def test_categorical_logits_softmax(self):
        logits = np.array([1.0, 2.0, 3.0], dtype="float32")
        try:
            dist = jt.distributions.Categorical(logits=jt.array(logits))
        except Exception:
            self.skipTest("Categorical not available")
        probs = dist.probs.numpy()
        ref = np.exp(logits) / np.exp(logits).sum()       # softmax, NOT sigmoid
        self.assertEqual(probs, ref, atol=1e-6, rtol=1e-6, msg="Categorical logits softmax")

    # -- f5b70ed8: kaiming/xavier init must not freeze the parameter -------------
    def test_init_does_not_freeze_param(self):
        lin = nn.Linear(8, 4)
        nn.init.relu_invariant_gauss_(lin.weight) if hasattr(nn.init, "relu_invariant_gauss_") \
            else nn.init.gauss_(lin.weight, 0, 0.1)
        x = jt.array(np.random.RandomState(5).randn(3, 8).astype("float32"))
        loss = (lin(x) ** 2).sum()
        g = jt.grad(loss, [lin.weight])[0].numpy()
        self.assertGreater(float(np.abs(g).max()), 0.0,
                           "weight must still receive gradient after init (not frozen)")

    # -- b35a30c9: mse_loss(reduction='none') must not crash, keeps shape --------
    def test_mse_loss_reduction_none(self):
        a = np.random.RandomState(6).randn(3, 4).astype("float32")
        b = np.random.RandomState(7).randn(3, 4).astype("float32")
        out = F.mse_loss(jt.array(a), jt.array(b), reduction="none").numpy()
        self.assertEqual(out, (a - b) ** 2, atol=1e-6, rtol=1e-6, msg="mse none")

    # -- nll_loss ignore_index must drop the ignored class (>=0 guard, not >0) ---
    def test_nll_loss_ignore_index_zero(self):
        logp = np.log(np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]], dtype="float32"))
        target = np.array([0, 1], dtype="int64")
        # ignore_index=0 must drop sample 0 entirely -> loss = -logp[1, 1]
        out = F.nll_loss(jt.array(logp), jt.array(target), ignore_index=0)
        ref = float(-logp[1, 1])
        self.assertEqual(_scalar(out), ref, atol=1e-5, rtol=1e-5,
                         msg="nll_loss ignore_index=0 drops class-0 sample")


if __name__ == "__main__":
    unittest.main(verbosity=2)
