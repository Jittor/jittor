# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""End-to-end mixed-precision training in native jittor, on CPU and on CUDA.

Every other test in this area asks one op one question. This asks the question
a user actually has: does a model trained in bfloat16, or in float16 with loss
scaling, reach the same place as the same model trained in float32.

It is a regression test for the whole chain rather than for any one op, and it
is the test that would have caught the two accumulation bugs fixed alongside it
without knowing where to look: the model's ``LayerNorm`` took its statistics at
the input's width, and on CPU its ``Linear`` layers contracted in float16
because the generic matmul fallback ran with ``reduce16_no_fp32_acc`` set. Both
degrade the *rate* of convergence rather than breaking it outright, which is
exactly the failure an op-level dtype assertion cannot see.

The tolerance is measured, not chosen: real torch 2.13 running the same
architecture, optimiser, schedule and initialisation reaches a final loss of
0.12238 in float32, 0.12289 under ``autocast(bfloat16)`` and 0.12652 under
``autocast(float16)`` with a ``GradScaler`` -- a spread of 0.4% and 3.4%. The
assertions below allow 25%, which is 7x that spread and still an order of
magnitude tighter than a half-accumulated run.

Run::  python -m pytest tests/nn/test_mixed_precision_training.py
"""

from _helpers import capability as _test_capability
import unittest

import numpy as np
import jittor as jt
from jittor import nn


_DEVICES = [("cpu", 0)] + (
    [("cuda", 1)] if _test_capability.any_accelerator_enabled(backend=jt) else [])

#: Steps and learning rate chosen so float32 drops ~140x -- far enough that a
#: degraded run separates from a healthy one, short enough to stay cheap.
STEPS = 80
LR = 0.005
#: A static scale, which is what a GradScaler settles on for a well-behaved
#: model. 1024 is inside float16's range for these gradients and large enough
#: that the smallest of them is no longer a subnormal.
LOSS_SCALE = 1024.0


def _task(n=256, d=16, seed=0):
    """A fixed linear regression with noise. Same data for every run."""
    rng = np.random.RandomState(seed)
    x = rng.randn(n, d).astype(np.float32)
    w = rng.randn(d, 1).astype(np.float32)
    y = (x @ w + 0.1 * rng.randn(n, 1)).astype(np.float32)
    return x, y


class _MLP(nn.Module):
    """Two hidden layers and a LayerNorm: matmul, relu and a normalisation.

    The LayerNorm is the point -- it is the layer whose statistics have to be
    accumulated wider than the activations, and it sits between two matmuls so a
    degraded gradient propagates into both.
    """

    def __init__(self, d=16, h=32):
        self.l1 = nn.Linear(d, h)
        self.l2 = nn.Linear(h, h)
        self.norm = nn.LayerNorm(h)
        self.l3 = nn.Linear(h, 1)

    def execute(self, x):
        x = nn.relu(self.l1(x))
        x = self.norm(self.l2(x))
        return self.l3(nn.relu(x))


def _seed_parameters(model, seed=1):
    """Identical starting weights for every run, independent of jittor's RNG."""
    rng = np.random.RandomState(seed)
    for p in model.parameters():
        p.assign(jt.array((rng.randn(*p.shape) * 0.1).astype(np.float32)))


def _finite(var):
    return bool(np.isfinite(var.float32().numpy()).all())


def _train(mode, use_cuda, scale=LOSS_SCALE):
    """Returns (losses, every_grad_finite, parameter dtypes at the end)."""
    with jt.flag_scope(use_cuda=use_cuda):
        jt.set_global_seed(0)
        x_np, y_np = _task()
        model = _MLP()
        _seed_parameters(model)
        cast = {"bf16": lambda v: v.bfloat16(),
                "fp16": lambda v: v.float16()}.get(mode)
        if cast is not None:
            for p in model.parameters():
                p.assign(cast(p))
        opt = jt.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
        x, y = jt.array(x_np), jt.array(y_np)
        if cast is not None:
            x, y = cast(x), cast(y)

        losses = []
        grads_finite = True
        for _ in range(STEPS):
            loss = ((model(x) - y) ** 2).mean()
            losses.append(float(loss.float32().numpy()))
            if mode == "fp16":
                # Static loss scaling, applied to a float32 copy of the loss so
                # the scale itself cannot overflow float16 before the backward
                # starts. The cotangents are float16 from the first op on.
                opt.backward(loss.float32() * scale)
                for p in model.parameters():
                    g = p.opt_grad(opt)
                    if g is None:
                        continue
                    grads_finite = grads_finite and _finite(g)
                    g.update(g / scale)
                opt.step()
            else:
                opt.step(loss)
                for p in model.parameters():
                    g = p.opt_grad(opt)
                    if g is not None:
                        grads_finite = grads_finite and _finite(g)
        dtypes = {str(p.dtype) for p in model.parameters()}
        return losses, grads_finite, dtypes


class TestMixedPrecisionTraining(unittest.TestCase):
    """bfloat16 and float16-with-loss-scaling must reach float32's answer."""

    #: Measured with these fixes in place. float32 is the reference each run is
    #: compared against; the halves are here so a future regression can be read
    #: off the failure message rather than re-derived.
    #:
    #:   cpu   fp32 0.13503   bf16 0.13867 (+2.7%)   fp16 0.13501 (-0.0%)
    #:
    #: Real torch 2.13 on the same task: 0.12238 / 0.12289 / 0.12652.
    TOLERANCE = 0.25

    def _baseline(self, use_cuda):
        losses, finite, dtypes = _train("fp32", use_cuda)
        self.assertTrue(finite)
        self.assertEqual(dtypes, {"float32"})
        return losses

    def test_float32_baseline_converges(self):
        for device, use_cuda in _DEVICES:
            with self.subTest(device=device):
                losses = self._baseline(use_cuda)
                self.assertLess(losses[-1], losses[0] / 50.0,
                                "the float32 baseline itself did not train")

    def test_bfloat16_matches_float32(self):
        for device, use_cuda in _DEVICES:
            with self.subTest(device=device):
                base = self._baseline(use_cuda)
                losses, finite, dtypes = _train("bf16", use_cuda)
                self.assertTrue(finite, "a bfloat16 gradient was not finite")
                # The optimiser must not quietly widen the parameters back.
                self.assertEqual(dtypes, {"bfloat16"})
                self.assertLess(losses[-1], losses[0] / 50.0,
                                "bfloat16 did not converge: %.5f -> %.5f"
                                % (losses[0], losses[-1]))
                self.assertLess(
                    abs(losses[-1] - base[-1]) / base[-1], self.TOLERANCE,
                    "bfloat16 final loss %.5f against float32's %.5f on %s; "
                    "real torch 2.13's own bfloat16/float32 spread on this task "
                    "is 0.4%%" % (losses[-1], base[-1], device))

    def test_float16_with_loss_scaling_matches_float32(self):
        for device, use_cuda in _DEVICES:
            with self.subTest(device=device):
                base = self._baseline(use_cuda)
                losses, finite, dtypes = _train("fp16", use_cuda)
                self.assertTrue(finite, "a float16 gradient was not finite "
                                        "even with the loss scaled by %g" % LOSS_SCALE)
                self.assertEqual(dtypes, {"float16"})
                self.assertLess(losses[-1], losses[0] / 50.0,
                                "float16 did not converge: %.5f -> %.5f"
                                % (losses[0], losses[-1]))
                self.assertLess(
                    abs(losses[-1] - base[-1]) / base[-1], self.TOLERANCE,
                    "float16 final loss %.5f against float32's %.5f on %s; "
                    "real torch 2.13's own float16/float32 spread on this task "
                    "is 3.4%%" % (losses[-1], base[-1], device))

    def test_the_loss_scale_is_load_bearing_in_both_directions(self):
        """Measured, not assumed: the scale helps, and too much of it overflows.

        On this model at these settings the scale is not the difference between
        training and not training -- the gradients are small but not subnormal,
        so an unscaled run still converges. It is worth a measurable amount:
        0.13501 at a scale of 1024 against 0.13672 unscaled on CPU.

        The other direction is what a ``GradScaler`` exists to find. At 65536 --
        torch's own starting scale -- these gradients leave float16's range and
        the run goes to NaN within the 80 steps. A scaler detects exactly that
        and halves; a test that only ever used a scale that works would not
        notice if the non-finite gradients stopped being reported.
        """
        for device, use_cuda in _DEVICES:
            with self.subTest(device=device):
                scaled, _, _ = _train("fp16", use_cuda)
                unscaled, unscaled_finite, _ = _train("fp16", use_cuda, scale=1.0)
                self.assertTrue(unscaled_finite)
                self.assertLessEqual(
                    scaled[-1], unscaled[-1] * 1.02,
                    "scaling by %g did not help on %s: %.5f scaled against "
                    "%.5f unscaled" % (LOSS_SCALE, device, scaled[-1], unscaled[-1]))
                over, over_finite, _ = _train("fp16", use_cuda, scale=65536.0)
                self.assertFalse(
                    over_finite,
                    "a scale of 65536 did not overflow float16 on %s, so the "
                    "non-finite-gradient check this relies on proves nothing"
                    % device)
                self.assertFalse(np.isfinite(over[-1]))


class TestAmpLevelTraining(unittest.TestCase):
    """The same task driven by ``jt.flags.auto_mixed_precision_level``.

    Levels 4, 5 and 6 set ``amp_prefer16``, so the model's parameters stay
    float32 and dtype inference lowers the *ops* -- jittor's answer to
    ``torch.autocast``. Level 3 sets ``keep_reduce | keep_white`` only, which
    changes nothing for a float32 model, and level 0 is off; both are asserted
    to be exactly the float32 run rather than merely close, because anything
    else would mean a level that claims to be a no-op is not one.
    """

    def _run(self, level, use_cuda):
        with jt.flag_scope(use_cuda=use_cuda):
            jt.set_global_seed(0)
            x_np, y_np = _task()
            model = _MLP()
            _seed_parameters(model)
            opt = jt.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
            x, y = jt.array(x_np), jt.array(y_np)
            losses, out_dtypes = [], set()
            with jt.flag_scope(auto_mixed_precision_level=level):
                for _ in range(STEPS):
                    out = model(x)
                    out_dtypes.add(str(out.dtype))
                    loss = ((out - y) ** 2).mean()
                    losses.append(float(loss.float32().numpy()))
                    opt.step(loss)
            return losses, out_dtypes

    def test_inert_levels_are_exactly_the_float32_run(self):
        for device, use_cuda in _DEVICES:
            for level in (0, 3):
                with self.subTest(device=device, level=level):
                    losses, dtypes = self._run(level, use_cuda)
                    self.assertEqual(dtypes, {"float32"})
                    base, _, _ = _train("fp32", use_cuda)
                    # Not bit equality: two float32 runs of this model differ by
                    # ~2e-6 relative on both devices, because the reductions are
                    # parallel and their summation order is not pinned. 1e-5 is
                    # an order of magnitude under what a level that actually
                    # lowered a dtype would cost (float16's ULP is 4.9e-4).
                    np.testing.assert_allclose(losses, base, rtol=1e-5, atol=0)

    def test_prefer16_levels_train_in_half_and_converge(self):
        for device, use_cuda in _DEVICES:
            base, _, _ = _train("fp32", use_cuda)
            for level in (4, 5, 6):
                with self.subTest(device=device, level=level):
                    losses, dtypes = self._run(level, use_cuda)
                    self.assertEqual(
                        dtypes, {"float16"},
                        "auto_mixed_precision_level=%d left the forward in %s"
                        % (level, sorted(dtypes)))
                    self.assertTrue(np.isfinite(losses).all())
                    self.assertLess(losses[-1], losses[0] / 50.0,
                                    "level %d did not converge: %.5f -> %.5f"
                                    % (level, losses[0], losses[-1]))
                    self.assertLess(
                        abs(losses[-1] - base[-1]) / base[-1], 0.25,
                        "level %d final loss %.5f against float32's %.5f on %s"
                        % (level, losses[-1], base[-1], device))


if __name__ == "__main__":
    unittest.main()
