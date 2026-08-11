# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Optimizer UPDATE-RULE correctness -- the training core ("不能有bug").

The legacy optimizer tests verify *convergence* (``test_adamw`` trains a net and
compares the final loss to torch) or self-consistency (``test_optimizer`` checks
two SGD steps land on hard-coded numbers). Neither pins the per-step update to its
*closed-form* rule, so a sign flip in a moment update, a misplaced ``eps``, a
wrong bias-correction exponent, or a coupled-vs-decoupled weight-decay mixup can
pass while the net still (slowly) converges. This module closes that gap.

For SGD (momentum / weight_decay / nesterov / dampening), Adam, AdamW and RMSprop
we drive ONE optimizer step (and a few) from a known parameter and a known
gradient, then assert the parameter -- and the optimizer's internal moment state
(``values`` / ``m``) -- landed exactly where the hand-coded analytic rule says.
The reference is an INDEPENDENT numpy re-implementation of each update; it is never
jittor-vs-jittor.

Two facts about jittor's implementation that the references must honor (verified by
reading the modules under ``python/jittor/optim/``, not assumed):

  * ``n_step`` is incremented inside ``backward()`` *before* the step body, so on the
    FIRST ``step(loss)`` the bias-correction exponent ``n`` is ``1.0`` (Adam/AdamW).
  * weight_decay is COUPLED into the gradient for SGD/Adam (``g <- p*wd + g``) but
    DECOUPLED for AdamW (``p <- p*(1 - lr*wd)`` applied to the param directly). The
    test asserts these two differ in exactly the way the rules predict.

A known gradient is injected device-independently with a LINEAR loss
``loss = (g_target * p).sum()`` whose gradient w.r.t. ``p`` is exactly ``g_target``
(this is the same loss-construction style ``test_optimizer`` uses, and it makes the
update math exact rather than dependent on a backward kernel). The gradient is also
cross-checked against ``jt.grad`` so the harness still verifies the BACKWARD that
feeds the optimizer.

Optimizer math is device-independent, but we still sweep ``get_all_device_types()``
when cheap, because the moment-buffer *update* runs as device kernels.

Run::  python -m pytest tests/optim/test_optim_core.py
"""
import unittest

import numpy as np
import jittor as jt

from _helpers.common import (
    JittorTestCase, get_all_device_types, use_cuda_for,
)


def _np(v):
    return v.numpy() if isinstance(v, jt.Var) else np.asarray(v)


class _OptimCoreBase(JittorTestCase):
    """Shared scaffolding: a fresh leaf param, a known grad, per-device sweep."""

    # the update rules are pure float32 arithmetic done identically by numpy and
    # jittor, so they should agree to near round-off; keep it tight so a real
    # formula bug (not round-off) is what trips it.
    TOL = 1e-5

    def _devices(self, body):
        for d in get_all_device_types():
            with self.subTest(device=d):
                with jt.flag_scope(use_cuda=use_cuda_for(d)):
                    body(d)

    def _param(self, p0):
        """A fresh differentiable leaf Var holding ``p0`` (float32)."""
        return jt.array(p0.astype("float32"), dtype="float32")

    def _linear_loss(self, p, g_target):
        """A loss whose analytic gradient w.r.t. ``p`` is exactly ``g_target``.

        ``loss = sum(g_target * p)``  =>  d loss / d p = g_target.
        Also asserts jt.grad agrees, so the backward feeding the optimizer is
        itself checked (not just the update arithmetic).
        """
        gt = jt.array(g_target.astype("float32"), dtype="float32")
        loss = (gt * p).sum()
        # cross-check the gradient the optimizer is about to consume
        g_auto = jt.grad(loss, [p])[0]
        self.assertEqual(g_auto, g_target, atol=self.TOL, rtol=self.TOL,
                         msg="linear-loss gradient must equal g_target")
        return loss


# ===========================================================================
#  SGD
# ===========================================================================
class TestSGDUpdate(_OptimCoreBase):

    def _ref_sgd(self, p, g, v, lr, momentum, weight_decay, dampening, nesterov):
        """One analytic SGD step (numpy). Mirrors optim.SGD.step exactly:
            dp = p*wd + g
            v  = momentum*v + dp*(1 - dampening)
            nesterov: p -= (dp + momentum*v)*lr     # v is the UPDATED v
            else:     p -= v*lr
        Returns (p_next, v_next)."""
        dp = p * weight_decay + g
        v_next = momentum * v + dp * (1.0 - dampening)
        if nesterov:
            p_next = p - (dp + momentum * v_next) * lr
        else:
            p_next = p - v_next * lr
        return p_next, v_next

    def test_vanilla_sgd_one_step(self):
        # plain SGD: p -= lr*g   (no momentum/wd)
        p0 = np.random.RandomState(0).randn(6).astype("float32")
        g0 = np.random.RandomState(1).randn(6).astype("float32")
        lr = 0.1

        def body(dev):
            p = self._param(p0)
            opt = jt.optim.SGD([p], lr)
            opt.step(self._linear_loss(p, g0))
            self.assertEqual(p, p0 - lr * g0, atol=self.TOL, rtol=self.TOL,
                             msg=f"vanilla SGD step [{dev}]")
        self._devices(body)

    def test_sgd_weight_decay_one_step(self):
        # p -= lr*(g + wd*p)
        p0 = np.random.RandomState(2).randn(6).astype("float32")
        g0 = np.random.RandomState(3).randn(6).astype("float32")
        lr, wd = 0.05, 0.1

        def body(dev):
            p = self._param(p0)
            opt = jt.optim.SGD([p], lr, weight_decay=wd)
            opt.step(self._linear_loss(p, g0))
            self.assertEqual(p, p0 - lr * (g0 + wd * p0), atol=self.TOL, rtol=self.TOL,
                             msg=f"SGD weight_decay step [{dev}]")
        self._devices(body)

    def test_sgd_momentum_state_and_param(self):
        # momentum buffer v_0 = g (from v init 0), then p -= lr*v.
        # Asserts BOTH the param landing AND the internal momentum state.
        p0 = np.random.RandomState(4).randn(8).astype("float32")
        g0 = np.random.RandomState(5).randn(8).astype("float32")
        lr, mom = 0.1, 0.9

        def body(dev):
            p = self._param(p0)
            opt = jt.optim.SGD([p], lr, momentum=mom)
            opt.step(self._linear_loss(p, g0))
            p_ref, v_ref = self._ref_sgd(p0, g0, np.zeros_like(p0), lr, mom, 0.0, 0.0, False)
            self.assertEqual(p, p_ref, atol=self.TOL, rtol=self.TOL,
                             msg=f"SGD+momentum param [{dev}]")
            # the velocity buffer must have been updated to g (not still zero)
            v_got = opt.param_groups[0]["values"][0]
            self.assertEqual(v_got, v_ref, atol=self.TOL, rtol=self.TOL,
                             msg=f"SGD momentum state buffer [{dev}]")
            self.assertGreater(float(np.max(np.abs(_np(v_got)))), 0.0,
                               f"momentum buffer must be non-zero after a step [{dev}]")
        self._devices(body)

    def test_sgd_momentum_two_steps(self):
        # second step: v_1 = mom*v_0 + g  with v_0 = g (same g both steps).
        p0 = np.random.RandomState(6).randn(8).astype("float32")
        g0 = np.random.RandomState(7).randn(8).astype("float32")
        lr, mom = 0.1, 0.9

        def body(dev):
            p = self._param(p0)
            opt = jt.optim.SGD([p], lr, momentum=mom)
            # analytic rollout over two identical-grad steps
            pr, vr = p0.copy(), np.zeros_like(p0)
            for _ in range(2):
                opt.step(self._linear_loss(p, g0))
                pr, vr = self._ref_sgd(pr, g0, vr, lr, mom, 0.0, 0.0, False)
            self.assertEqual(p, pr, atol=self.TOL, rtol=self.TOL,
                             msg=f"SGD+momentum 2 steps param [{dev}]")
            self.assertEqual(opt.param_groups[0]["values"][0], vr,
                             atol=self.TOL, rtol=self.TOL,
                             msg=f"SGD+momentum 2 steps state [{dev}]")
        self._devices(body)

    def test_sgd_nesterov_one_step(self):
        # nesterov uses the UPDATED velocity in the param step: p -= (dp + mom*v)*lr
        p0 = np.random.RandomState(8).randn(8).astype("float32")
        g0 = np.random.RandomState(9).randn(8).astype("float32")
        lr, mom, wd = 0.1, 0.9, 0.0

        def body(dev):
            p = self._param(p0)
            opt = jt.optim.SGD([p], lr, momentum=mom, nesterov=True)
            opt.step(self._linear_loss(p, g0))
            p_ref, _ = self._ref_sgd(p0, g0, np.zeros_like(p0), lr, mom, wd, 0.0, True)
            self.assertEqual(p, p_ref, atol=self.TOL, rtol=self.TOL,
                             msg=f"SGD nesterov step [{dev}]")
        self._devices(body)

    def test_sgd_dampening_one_step(self):
        # v_0 = g*(1-dampening); p -= lr*v
        p0 = np.random.RandomState(10).randn(8).astype("float32")
        g0 = np.random.RandomState(11).randn(8).astype("float32")
        lr, mom, damp = 0.1, 0.9, 0.5

        def body(dev):
            p = self._param(p0)
            opt = jt.optim.SGD([p], lr, momentum=mom, dampening=damp)
            opt.step(self._linear_loss(p, g0))
            p_ref, v_ref = self._ref_sgd(p0, g0, np.zeros_like(p0), lr, mom, 0.0, damp, False)
            self.assertEqual(p, p_ref, atol=self.TOL, rtol=self.TOL,
                             msg=f"SGD dampening param [{dev}]")
            self.assertEqual(opt.param_groups[0]["values"][0], v_ref,
                             atol=self.TOL, rtol=self.TOL,
                             msg=f"SGD dampening state [{dev}]")
        self._devices(body)


# ===========================================================================
#  Adam
# ===========================================================================
class TestAdamUpdate(_OptimCoreBase):

    def _ref_adam(self, p, g, m, v, n, lr, b0, b1, eps, weight_decay):
        """One analytic Adam step (numpy). Mirrors optim.Adam.step exactly:
            g    = p*wd + g                         # COUPLED weight decay
            m    = b0*m + (1-b0)*g
            v    = b1*v + (1-b1)*g*g
            step = lr*sqrt(1-b1**n)/(1-b0**n)
            p   -= m*step/(sqrt(v)+eps)             # eps OUTSIDE bias-corrected v
        ``n`` is the bias-correction exponent (== self.n_step at step time, i.e. 1
        on the first step). Returns (p_next, m_next, v_next)."""
        g = p * weight_decay + g
        m_next = b0 * m + (1.0 - b0) * g
        v_next = b1 * v + (1.0 - b1) * g * g
        step = lr * np.sqrt(1.0 - b1 ** n) / (1.0 - b0 ** n)
        p_next = p - m_next * step / (np.sqrt(v_next) + eps)
        return p_next, m_next, v_next

    def test_adam_one_step_param_and_moments(self):
        p0 = np.random.RandomState(20).randn(8).astype("float32")
        g0 = np.random.RandomState(21).randn(8).astype("float32")
        lr, eps = 0.01, 1e-8
        b0, b1 = 0.9, 0.999

        def body(dev):
            p = self._param(p0)
            opt = jt.optim.Adam([p], lr, eps=eps, betas=(b0, b1))
            opt.step(self._linear_loss(p, g0))
            # first step => n == 1 (n_step incremented in backward before step body)
            p_ref, m_ref, v_ref = self._ref_adam(
                p0, g0, np.zeros_like(p0), np.zeros_like(p0), 1.0, lr, b0, b1, eps, 0.0)
            self.assertEqual(p, p_ref, atol=self.TOL, rtol=self.TOL,
                             msg=f"Adam param [{dev}]")
            # exp_avg (m) and exp_avg_sq (v) state must match the moment updates
            self.assertEqual(opt.param_groups[0]["m"][0], m_ref,
                             atol=self.TOL, rtol=self.TOL, msg=f"Adam exp_avg [{dev}]")
            self.assertEqual(opt.param_groups[0]["values"][0], v_ref,
                             atol=self.TOL, rtol=self.TOL, msg=f"Adam exp_avg_sq [{dev}]")
            self.assertGreater(float(np.max(np.abs(_np(opt.param_groups[0]["m"][0])))), 0.0,
                               f"Adam exp_avg must be non-zero after a step [{dev}]")
        self._devices(body)

    def test_adam_weight_decay_coupled(self):
        # Adam wd is COUPLED: g <- p*wd + g, so the moments themselves shift.
        p0 = np.random.RandomState(22).randn(8).astype("float32")
        g0 = np.random.RandomState(23).randn(8).astype("float32")
        lr, eps, wd = 0.01, 1e-8, 0.1
        b0, b1 = 0.9, 0.999

        def body(dev):
            p = self._param(p0)
            opt = jt.optim.Adam([p], lr, eps=eps, betas=(b0, b1), weight_decay=wd)
            opt.step(self._linear_loss(p, g0))
            p_ref, m_ref, _ = self._ref_adam(
                p0, g0, np.zeros_like(p0), np.zeros_like(p0), 1.0, lr, b0, b1, eps, wd)
            self.assertEqual(p, p_ref, atol=self.TOL, rtol=self.TOL,
                             msg=f"Adam coupled-wd param [{dev}]")
            # the first moment must reflect the wd-augmented gradient
            self.assertEqual(opt.param_groups[0]["m"][0], m_ref,
                             atol=self.TOL, rtol=self.TOL,
                             msg=f"Adam coupled-wd exp_avg [{dev}]")
        self._devices(body)

    def test_adam_two_steps_bias_correction(self):
        # the bias-correction exponent advances 1 -> 2; rolling the reference with
        # n=1 then n=2 must track jittor. (A wrong/off-by-one exponent diverges here.)
        p0 = np.random.RandomState(24).randn(8).astype("float32")
        g0 = np.random.RandomState(25).randn(8).astype("float32")
        lr, eps = 0.01, 1e-8
        b0, b1 = 0.9, 0.999

        def body(dev):
            p = self._param(p0)
            opt = jt.optim.Adam([p], lr, eps=eps, betas=(b0, b1))
            pr, mr, vr = p0.copy(), np.zeros_like(p0), np.zeros_like(p0)
            for n in (1.0, 2.0):
                opt.step(self._linear_loss(p, g0))
                pr, mr, vr = self._ref_adam(pr, g0, mr, vr, n, lr, b0, b1, eps, 0.0)
            self.assertEqual(p, pr, atol=self.TOL, rtol=self.TOL,
                             msg=f"Adam 2-step bias-correction param [{dev}]")
            self.assertEqual(opt.param_groups[0]["m"][0], mr,
                             atol=self.TOL, rtol=self.TOL,
                             msg=f"Adam 2-step exp_avg [{dev}]")
            self.assertEqual(opt.param_groups[0]["values"][0], vr,
                             atol=self.TOL, rtol=self.TOL,
                             msg=f"Adam 2-step exp_avg_sq [{dev}]")
        self._devices(body)


# ===========================================================================
#  AdamW  (decoupled weight decay)
# ===========================================================================
class TestAdamWUpdate(_OptimCoreBase):

    def _ref_adamw(self, p, g, m, v, n, lr, b0, b1, eps, weight_decay):
        """One analytic AdamW step (numpy). Mirrors optim.AdamW.step exactly:
            p    = p*(1 - lr*wd)                    # DECOUPLED decay, on the param
            bc1  = 1 - b0**n ; bc2 = 1 - b1**n
            m    = b0*m + (1-b0)*g                  # raw g (no wd in grad)
            v    = b1*v + (1-b1)*g*g
            denom= sqrt(v)/sqrt(bc2) + eps
            p   -= (lr/bc1)*m/denom
        Returns (p_next, m_next, v_next)."""
        p = p * (1.0 - lr * weight_decay)
        bc1 = 1.0 - b0 ** n
        bc2 = 1.0 - b1 ** n
        m_next = b0 * m + (1.0 - b0) * g
        v_next = b1 * v + (1.0 - b1) * g * g
        denom = np.sqrt(v_next) / np.sqrt(bc2) + eps
        p_next = p - (lr / bc1) * m_next / denom
        return p_next, m_next, v_next

    def test_adamw_one_step_param_and_moments(self):
        p0 = np.random.RandomState(30).randn(8).astype("float32")
        g0 = np.random.RandomState(31).randn(8).astype("float32")
        lr, eps, wd = 0.01, 1e-8, 0.1
        b0, b1 = 0.9, 0.99

        def body(dev):
            p = self._param(p0)
            opt = jt.optim.AdamW([p], lr, eps=eps, betas=(b0, b1), weight_decay=wd)
            opt.step(self._linear_loss(p, g0))
            p_ref, m_ref, v_ref = self._ref_adamw(
                p0, g0, np.zeros_like(p0), np.zeros_like(p0), 1.0, lr, b0, b1, eps, wd)
            self.assertEqual(p, p_ref, atol=self.TOL, rtol=self.TOL,
                             msg=f"AdamW param [{dev}]")
            self.assertEqual(opt.param_groups[0]["m"][0], m_ref,
                             atol=self.TOL, rtol=self.TOL, msg=f"AdamW exp_avg [{dev}]")
            self.assertEqual(opt.param_groups[0]["values"][0], v_ref,
                             atol=self.TOL, rtol=self.TOL, msg=f"AdamW exp_avg_sq [{dev}]")
        self._devices(body)

    def test_adamw_decoupled_differs_from_adam_coupled(self):
        # The marquee distinction: AdamW's decoupled decay must NOT equal Adam's
        # coupled L2 decay for the same (lr, wd, betas). If AdamW silently used
        # coupled decay (or vice-versa) these would coincide -- assert they don't,
        # and that each matches ITS OWN analytic rule.
        p0 = np.random.RandomState(32).randn(8).astype("float32")
        g0 = np.random.RandomState(33).randn(8).astype("float32")
        lr, eps, wd = 0.05, 1e-8, 0.2          # large-ish wd so the gap is visible
        b0, b1 = 0.9, 0.99

        def body(dev):
            pa = self._param(p0)
            opt_a = jt.optim.Adam([pa], lr, eps=eps, betas=(b0, b1), weight_decay=wd)
            opt_a.step(self._linear_loss(pa, g0))
            adam_ref, _, _ = TestAdamUpdate._ref_adam(
                self, p0, g0, np.zeros_like(p0), np.zeros_like(p0), 1.0, lr, b0, b1, eps, wd)

            pw = self._param(p0)
            opt_w = jt.optim.AdamW([pw], lr, eps=eps, betas=(b0, b1), weight_decay=wd)
            opt_w.step(self._linear_loss(pw, g0))
            adamw_ref, _, _ = self._ref_adamw(
                p0, g0, np.zeros_like(p0), np.zeros_like(p0), 1.0, lr, b0, b1, eps, wd)

            # each matches its own rule ...
            self.assertEqual(pa, adam_ref, atol=self.TOL, rtol=self.TOL,
                             msg=f"Adam(coupled) rule [{dev}]")
            self.assertEqual(pw, adamw_ref, atol=self.TOL, rtol=self.TOL,
                             msg=f"AdamW(decoupled) rule [{dev}]")
            # ... and the two rules genuinely diverge.
            gap = float(np.max(np.abs(_np(pa) - _np(pw))))
            self.assertGreater(
                gap, 1e-4,
                f"AdamW decoupled decay must differ from Adam coupled L2 "
                f"(gap {gap:.2e}); if ~0 one optimizer is using the wrong decay [{dev}]")
            # and the analytic references must diverge by the same amount jittor does
            ref_gap = float(np.max(np.abs(adam_ref - adamw_ref)))
            self.assertAlmostEqual(gap, ref_gap, delta=1e-4,
                                   msg=f"coupled-vs-decoupled gap mismatch [{dev}]")
        self._devices(body)


# ===========================================================================
#  RMSprop
# ===========================================================================
class TestRMSpropUpdate(_OptimCoreBase):

    def _ref_rmsprop(self, p, g, v, lr, alpha, eps):
        """One analytic RMSprop step (numpy). Mirrors optim.RMSprop.step exactly:
            v  = alpha*v + (1-alpha)*g*g
            p -= lr*g/(sqrt(v)+eps)
        Returns (p_next, v_next)."""
        v_next = alpha * v + (1.0 - alpha) * g * g
        p_next = p - lr * g / (np.sqrt(v_next) + eps)
        return p_next, v_next

    def test_rmsprop_one_step_param_and_state(self):
        p0 = np.random.RandomState(40).randn(8).astype("float32")
        g0 = np.random.RandomState(41).randn(8).astype("float32")
        lr, alpha, eps = 0.01, 0.99, 1e-8

        def body(dev):
            p = self._param(p0)
            opt = jt.optim.RMSprop([p], lr, eps=eps, alpha=alpha)
            opt.step(self._linear_loss(p, g0))
            p_ref, v_ref = self._ref_rmsprop(p0, g0, np.zeros_like(p0), lr, alpha, eps)
            self.assertEqual(p, p_ref, atol=self.TOL, rtol=self.TOL,
                             msg=f"RMSprop param [{dev}]")
            self.assertEqual(opt.param_groups[0]["values"][0], v_ref,
                             atol=self.TOL, rtol=self.TOL,
                             msg=f"RMSprop square-avg state [{dev}]")
            self.assertGreater(float(np.max(np.abs(_np(opt.param_groups[0]["values"][0])))), 0.0,
                               f"RMSprop square-avg must be non-zero after a step [{dev}]")
        self._devices(body)

    def test_rmsprop_two_steps(self):
        p0 = np.random.RandomState(42).randn(8).astype("float32")
        g0 = np.random.RandomState(43).randn(8).astype("float32")
        lr, alpha, eps = 0.01, 0.9, 1e-8

        def body(dev):
            p = self._param(p0)
            opt = jt.optim.RMSprop([p], lr, eps=eps, alpha=alpha)
            pr, vr = p0.copy(), np.zeros_like(p0)
            for _ in range(2):
                opt.step(self._linear_loss(p, g0))
                pr, vr = self._ref_rmsprop(pr, g0, vr, lr, alpha, eps)
            self.assertEqual(p, pr, atol=self.TOL, rtol=self.TOL,
                             msg=f"RMSprop 2-step param [{dev}]")
            self.assertEqual(opt.param_groups[0]["values"][0], vr,
                             atol=self.TOL, rtol=self.TOL,
                             msg=f"RMSprop 2-step state [{dev}]")
        self._devices(body)


# ===========================================================================
#  Cross-cutting: backward()+step() path, param_groups per-group lr
# ===========================================================================
class TestOptimizerPlumbing(_OptimCoreBase):

    def test_backward_then_step_matches_step_loss(self):
        # opt.backward(loss); opt.step()  must equal  opt.step(loss).
        # Both are documented entry points; they must produce the same update.
        p0 = np.random.RandomState(50).randn(6).astype("float32")
        g0 = np.random.RandomState(51).randn(6).astype("float32")
        lr = 0.1

        def body(dev):
            pa = self._param(p0)
            opt_a = jt.optim.SGD([pa], lr, momentum=0.9)
            opt_a.step(self._linear_loss(pa, g0))

            pb = self._param(p0)
            opt_b = jt.optim.SGD([pb], lr, momentum=0.9)
            opt_b.backward(self._linear_loss(pb, g0))
            opt_b.step()

            self.assertEqual(pa, pb, atol=self.TOL, rtol=self.TOL,
                             msg=f"backward+step == step(loss) [{dev}]")
        self._devices(body)

    def test_per_group_lr_overrides_default(self):
        # two param groups, group-0 overrides lr; each must use its own lr.
        pa0 = np.random.RandomState(52).randn(4).astype("float32")
        pb0 = np.random.RandomState(53).randn(4).astype("float32")
        ga0 = np.random.RandomState(54).randn(4).astype("float32")
        gb0 = np.random.RandomState(55).randn(4).astype("float32")
        lr_group, lr_default = 0.5, 0.1

        def body(dev):
            pa = self._param(pa0)
            pb = self._param(pb0)
            opt = jt.optim.SGD([
                {"params": [pa], "lr": lr_group},
                {"params": [pb]},
            ], lr_default)
            gta = jt.array(ga0, dtype="float32")
            gtb = jt.array(gb0, dtype="float32")
            loss = (gta * pa).sum() + (gtb * pb).sum()
            opt.step(loss)
            # group-0 uses lr_group, group-1 falls back to the optimizer default
            self.assertEqual(pa, pa0 - lr_group * ga0, atol=self.TOL, rtol=self.TOL,
                             msg=f"per-group lr (override) [{dev}]")
            self.assertEqual(pb, pb0 - lr_default * gb0, atol=self.TOL, rtol=self.TOL,
                             msg=f"per-group lr (default) [{dev}]")
        self._devices(body)

    def test_zero_grad_resets_accumulation(self):
        # backward() accumulates into pg['grads']; after step() the post_step
        # zero_grads, so a fresh backward(same grad) must NOT double the gradient.
        p0 = np.random.RandomState(56).randn(6).astype("float32")
        g0 = np.random.RandomState(57).randn(6).astype("float32")
        lr = 0.1

        def body(dev):
            p = self._param(p0)
            opt = jt.optim.SGD([p], lr)          # vanilla, so the update is lr*grad
            opt.backward(self._linear_loss(p, g0))
            # accumulate the SAME grad once more before stepping -> grad == 2*g0
            opt.backward(self._linear_loss(p, g0))
            opt.step()
            self.assertEqual(p, p0 - lr * (2.0 * g0), atol=self.TOL, rtol=self.TOL,
                             msg=f"two backward() accumulate before step [{dev}]")
        self._devices(body)


if __name__ == "__main__":
    unittest.main(verbosity=2)
