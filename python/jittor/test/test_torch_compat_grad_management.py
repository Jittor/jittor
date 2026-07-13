"""Gradient clipping and AMP-scaler correctness/synchronization regressions."""

import math
import unittest

import numpy as np
import jittor as jt
from jittor.torch_compat import _GradScaler, _clip_grad_norm_device


_DEVICES = [("cpu", 0)] + ([ ("cuda", 1) ] if jt.has_cuda else [])


def both_devices(fn):
    for name, use_cuda in _DEVICES:
        with jt.flag_scope(use_cuda=use_cuda):
            fn(name)


class _FakeOptimizer:
    def __init__(self, grads):
        self.param_groups = [{"grads": grads}]
        self.steps = 0

    def step(self, *args, **kwargs):
        self.steps += 1
        return self.steps


class TestGradientManagement(unittest.TestCase):
    def test_autograd_grad_unused_semantics(self):
        def make_graph():
            used = jt.array(np.array([1.0, 2.0], dtype="float32"))
            unused = jt.array(np.array([3.0, 4.0], dtype="float32"))
            return used, unused, (used * used).sum()

        used, unused, loss = make_graph()
        grads = jt.autograd.grad(loss, (used, unused), allow_unused=True)
        self.assertIsNotNone(grads[0])
        self.assertIsNone(grads[1])

        used, unused, loss = make_graph()
        with self.assertRaisesRegex(RuntimeError, "allow_unused=True"):
            jt.autograd.grad(loss, (used, unused))

        used, unused, loss = make_graph()
        grads = jt.autograd.grad(
            loss, (used, unused), materialize_grads=True)
        np.testing.assert_array_equal(grads[1].numpy(), np.zeros(2, dtype="float32"))

        used, unused, loss = make_graph()
        with self.assertRaisesRegex(ValueError, "allow_unused"):
            jt.autograd.grad(
                loss, (used, unused), allow_unused=False,
                materialize_grads=True)

        used, unused, loss = make_graph()
        grads = jt.autograd.grad(
            loss, (used, unused), create_graph=True,
            materialize_grads=True)
        self.assertTrue(grads[1].requires_grad)

    def test_clip_grad_norm_values_and_no_item(self):
        def body(dev):
            for p, expected in ((1.0, 10.0), (2.0, math.sqrt(34.0)),
                                (float("inf"), 4.0), (3.0, 118.0 ** (1.0 / 3.0))):
                grads = [jt.array(np.array([3.0, 4.0], dtype="float32")),
                         jt.array(np.array([0.0, -3.0], dtype="float32"))]
                calls = []
                original = jt.Var.item

                def counted(var, *args, **kwargs):
                    calls.append(1)
                    return original(var, *args, **kwargs)

                jt.Var.item = counted
                try:
                    total = _clip_grad_norm_device(grads, 1.0, p)
                    values = jt.fetch_sync([total] + grads)
                finally:
                    jt.Var.item = original
                self.assertEqual(len(calls), 0, f"clip_grad_norm_ synced on {dev}, p={p}")
                self.assertAlmostEqual(float(values[0].reshape(-1)[0]), expected, places=5)
                clipped = np.concatenate([values[1].reshape(-1), values[2].reshape(-1)])
                if p == float("inf"):
                    got_norm = np.abs(clipped).max()
                else:
                    got_norm = np.linalg.norm(clipped, ord=p)
                self.assertLessEqual(float(got_norm), 1.00001)

        both_devices(body)

    def test_clip_grad_norm_nonfinite_error(self):
        def body(dev):
            grad = jt.array(np.array([1.0, np.inf], dtype="float32"))
            with self.assertRaisesRegex(RuntimeError, "non-finite"):
                _clip_grad_norm_device([grad], 1.0, 2.0, error_if_nonfinite=True)

        both_devices(body)

    def test_clip_grad_norm_nonfinite_propagates_when_allowed(self):
        def body(dev):
            for value in (np.inf, np.nan):
                grad = jt.array(np.array([1.0, value], dtype="float32"))
                total = _clip_grad_norm_device(
                    [grad], 1.0, 2.0, error_if_nonfinite=False)
                total_np, grad_np = jt.fetch_sync([total, grad])
                self.assertFalse(np.isfinite(total_np).all(), f"total {value} on {dev}")
                if np.isnan(value):
                    self.assertTrue(np.isnan(grad_np).all(), f"NaN propagation on {dev}")
                else:
                    self.assertEqual(float(grad_np[0]), 0.0, f"Inf finite entry on {dev}")
                    self.assertTrue(np.isnan(grad_np[1]), f"Inf*0 on {dev}")

        both_devices(body)

    def test_clip_grad_norm_zero_and_negative_inf(self):
        def body(dev):
            for p, expected in ((0.0, 2.0), (float("-inf"), 0.0)):
                grads = [jt.array(np.array([3.0, 0.0], dtype="float32")),
                         jt.array(np.array([4.0], dtype="float32"))]
                total = _clip_grad_norm_device(grads, float("inf"), p)
                got = float(total.item())
                self.assertEqual(got, expected, f"p={p} on {dev}")

        both_devices(body)

    def test_grad_scaler_uses_one_item_and_skips_inf_step(self):
        def body(dev):
            finite = [jt.array(np.array([8.0, -4.0], dtype="float32")),
                      jt.array(np.array([2.0], dtype="float32"))]
            opt = _FakeOptimizer(finite)
            scaler = _GradScaler(init_scale=2.0)
            calls = []
            original = jt.Var.item

            def counted(var, *args, **kwargs):
                calls.append(1)
                return original(var, *args, **kwargs)

            jt.Var.item = counted
            try:
                scaler.unscale_(opt)
            finally:
                jt.Var.item = original
            self.assertEqual(len(calls), 1, f"GradScaler item count on {dev}")
            self.assertFalse(scaler._found_inf)
            self.assertEqual(scaler.step(opt), 1)
            np.testing.assert_allclose(finite[0].numpy(), [4.0, -2.0], atol=1e-6)

            bad = _FakeOptimizer([
                jt.array(np.array([2.0, np.inf], dtype="float32")),
                jt.array(np.array([np.nan], dtype="float32")),
            ])
            scaler = _GradScaler(init_scale=2.0)
            scaler.unscale_(bad)
            self.assertTrue(scaler._found_inf)
            self.assertIsNone(scaler.step(bad))
            self.assertEqual(bad.steps, 0)

        both_devices(body)


if __name__ == "__main__":
    unittest.main()
