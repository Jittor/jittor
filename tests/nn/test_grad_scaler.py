# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""``jt.amp.GradScaler``: the loss scaler float16 training needs.

Before this class existed, native fp16 training meant writing the scaling by
hand at every call site -- multiply the loss, walk ``opt_grad`` for every
parameter, divide, and hope no gradient overflowed, because there was nowhere
to put the skip. ``tests/nn/test_mixed_precision_training.py`` still does that
by hand on purpose: it pins the *numerics* of a fixed scale, independent of
this class. What is checked here is the policy -- that the scale moves the way
it is supposed to, and that a step with a non-finite gradient does not happen.
"""
import unittest

import numpy as np

import jittor as jt
from jittor import nn

from _helpers import capability as _test_capability

#: The devices this box can actually run on, in the shape the sibling
#: end-to-end test uses: (name, use_cuda).
_DEVICES = [("cpu", 0)] + (
    [("cuda", 1)] if _test_capability.any_accelerator_enabled(backend=jt) else [])


class _Tiny(nn.Module):
    def __init__(self, d=8):
        self.l1 = nn.Linear(d, d)
        self.l2 = nn.Linear(d, 1)

    def execute(self, x):
        return self.l2(nn.relu(self.l1(x)))


def _model_and_optimizer(dtype="float32", seed=3):
    jt.set_global_seed(0)
    model = _Tiny()
    rng = np.random.RandomState(seed)
    for p in model.parameters():
        value = (rng.randn(*p.shape) * 0.1).astype(np.float32)
        p.assign(jt.array(value) if dtype == "float32"
                 else jt.array(value).cast(dtype))
    return model, jt.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)


class TestGradScalerPolicy(unittest.TestCase):
    """The scale, and what moves it. No device needed."""

    def test_a_disabled_scaler_is_the_identity(self):
        scaler = jt.amp.GradScaler(init_scale=1024.0, enabled=False)
        x = jt.ones(3)
        self.assertIs(scaler.scale(x), x)
        self.assertEqual(scaler.get_scale(), 1.0)
        # torch returns {} for a disabled scaler, and refuses to load one.
        self.assertEqual(scaler.state_dict(), {})

    def test_scale_walks_containers(self):
        scaler = jt.amp.GradScaler(init_scale=4.0)
        a, b = jt.ones(2), jt.full((2,), 3.0)
        scaled = scaler.scale([a, b])
        self.assertIsInstance(scaled, list)
        np.testing.assert_allclose(scaled[0].numpy(), [4.0, 4.0])
        np.testing.assert_allclose(scaled[1].numpy(), [12.0, 12.0])
        with self.assertRaises(ValueError):
            scaler.scale(object())

    def test_growth_needs_a_clean_run_of_growth_interval(self):
        scaler = jt.amp.GradScaler(init_scale=2.0, growth_factor=2.0,
                                   growth_interval=3)
        for _ in range(2):
            scaler.update()
            self.assertEqual(scaler.get_scale(), 2.0)
        scaler.update()
        self.assertEqual(scaler.get_scale(), 4.0)

    def test_backoff_resets_the_run_and_never_goes_below_one(self):
        scaler = jt.amp.GradScaler(init_scale=2.0, backoff_factor=0.5,
                                   growth_interval=2)
        scaler._found_inf = True
        scaler.update()
        self.assertEqual(scaler.get_scale(), 1.0)
        # and again: a scale below 1 would shrink the gradients, which is the
        # opposite of what the scaler is for.
        scaler._found_inf = True
        scaler.update()
        self.assertEqual(scaler.get_scale(), 1.0)
        # the clean-run counter restarted, so one clean update is not enough
        scaler.update()
        self.assertEqual(scaler.get_scale(), 1.0)
        scaler.update()
        self.assertEqual(scaler.get_scale(), 2.0)

    def test_state_dict_round_trips_and_carries_the_policy(self):
        a = jt.amp.GradScaler(init_scale=512.0, growth_factor=4.0,
                              backoff_factor=0.25, growth_interval=7)
        a.update()
        b = jt.amp.GradScaler()
        b.load_state_dict(a.state_dict())
        self.assertEqual(b.state_dict(), a.state_dict())
        self.assertEqual(b.get_growth_factor(), 4.0)
        self.assertEqual(b.get_backoff_factor(), 0.25)
        self.assertEqual(b.get_growth_interval(), 7)
        with self.assertRaises(RuntimeError):
            jt.amp.GradScaler().load_state_dict({})


class TestGradScalerStep(unittest.TestCase):
    """What ``step`` does to the optimizer, on whichever device is selected."""

    def test_step_unscales_the_gradients_it_hands_the_optimizer(self):
        for name, use_cuda in _DEVICES:
            with self.subTest(device=name), jt.flag_scope(use_cuda=use_cuda):
                model, opt = _model_and_optimizer()
                scaler = jt.amp.GradScaler(init_scale=128.0)
                x = jt.array(
                    np.random.RandomState(0).randn(4, 8).astype("float32"))
                opt.backward(scaler.scale((model(x) ** 2).mean()))
                scaled = {id(p): p.opt_grad(opt).numpy().copy()
                          for p in model.parameters()}
                scaler.unscale_(opt)
                for p in model.parameters():
                    np.testing.assert_allclose(
                        p.opt_grad(opt).numpy(), scaled[id(p)] / 128.0,
                        rtol=1e-6, atol=1e-7)

    def test_a_non_finite_gradient_skips_the_step_and_backs_off(self):
        for name, use_cuda in _DEVICES:
            with self.subTest(device=name), jt.flag_scope(use_cuda=use_cuda):
                model, opt = _model_and_optimizer()
                scaler = jt.amp.GradScaler(init_scale=1024.0,
                                           backoff_factor=0.5)
                x = jt.array(
                    np.random.RandomState(1).randn(4, 8).astype("float32"))
                opt.backward((model(x) ** 2).mean())
                # Poison one gradient the way an overflow would.
                g = model.parameters()[0].opt_grad(opt)
                g.update(g + np.float32("inf"))
                before = [p.numpy().copy() for p in model.parameters()]
                self.assertIsNone(scaler.step(opt))
                for p, was in zip(model.parameters(), before):
                    np.testing.assert_array_equal(p.numpy(), was)
                scaler.update()
                self.assertEqual(scaler.get_scale(), 512.0)

    def test_a_clean_step_updates_the_parameters(self):
        for name, use_cuda in _DEVICES:
            with self.subTest(device=name), jt.flag_scope(use_cuda=use_cuda):
                model, opt = _model_and_optimizer()
                scaler = jt.amp.GradScaler(init_scale=256.0)
                x = jt.array(
                    np.random.RandomState(2).randn(4, 8).astype("float32"))
                opt.backward(scaler.scale((model(x) ** 2).mean()))
                before = [p.numpy().copy() for p in model.parameters()]
                scaler.step(opt)
                jt.sync_all()
                moved = any(not np.array_equal(p.numpy(), was)
                            for p, was in zip(model.parameters(), before))
                self.assertTrue(moved,
                                "a clean step left every parameter alone")


class TestGradScalerTraining(unittest.TestCase):
    """The reason the class exists: fp16 training that converges."""

    STEPS = 40

    def _train(self, dtype, scaler):
        model, opt = _model_and_optimizer(dtype)
        rng = np.random.RandomState(7)
        x_np = rng.randn(32, 8).astype("float32")
        y_np = (x_np[:, :1] * 0.5 + 0.1).astype("float32")
        x, y = jt.array(x_np), jt.array(y_np)
        if dtype != "float32":
            x, y = x.cast(dtype), y.cast(dtype)
        losses = []
        for _ in range(self.STEPS):
            loss = ((model(x) - y) ** 2).mean()
            losses.append(float(loss.float32().numpy()))
            opt.backward(scaler.scale(loss.float32()))
            scaler.step(opt)
            scaler.update()
        return losses

    def test_float16_with_the_scaler_tracks_float32(self):
      for name, use_cuda in _DEVICES:
       with self.subTest(device=name), jt.flag_scope(use_cuda=use_cuda):
        reference = self._train("float32", jt.amp.GradScaler(enabled=False))
        half = self._train("float16", jt.amp.GradScaler(init_scale=2.0 ** 12))
        self.assertLess(half[-1], half[0],
                        "fp16 run did not converge at all")
        # Same task, same initialisation: the fp16 run should land within a
        # few percent of the fp32 one. Without scaling the small gradients
        # flush to zero and the loss plateaus instead.
        self.assertLess(abs(half[-1] - reference[-1]),
                        0.1 * max(reference[-1], 1e-6) + 1e-3,
                        "fp16 final loss %.6f vs fp32 %.6f"
                        % (half[-1], reference[-1]))

if __name__ == "__main__":
    unittest.main()
