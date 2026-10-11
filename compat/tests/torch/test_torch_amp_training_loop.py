"""The ordinary AMP training loop, end to end: autocast, GradScaler, AdamW.

Two defects sat on this path, one under the other.

* Reading the gradients of ``nn.Sequential(Linear, GELU, Linear)`` after an
  autocast forward + backward died in the planner with ``exec_plan.cc: [check
  failed: queue.size() == roots.size()]``. ``fuse_op_limit`` bounds a
  half-precision fused group, and the greedy bound could union two ops while
  refusing an op on the path between them, which leaves the fused groups in a
  cycle (tests/core/test_fuser.py pins the planner half natively).
* With that fixed, the CPU loop went NaN on its second step: a float32 weight
  came back from backward with a float16 ``.grad``, and AdamW's
  ``(1 - beta2) * g * g`` underflows in float16. torch gives a leaf a grad of
  the leaf's own dtype.

The loop is compared against the same loop with autocast off. Half precision
does not reproduce float32, so the tolerance is loose; what it catches is a
loop that diverges, goes NaN, or stops training.
"""

import unittest

import numpy as np

import jittor as jt
import torch
from torch import nn

from _helpers import common as cu
from _helpers.device_types import instantiate_device_type_tests


def _region(device):
    return "cpu" if device == "cpu" else "cuda"


def _half_dtypes(device):
    # CPU autocast is bfloat16 in torch; jittor's register approximates it.
    return [torch.bfloat16] if device == "cpu" else [torch.float16, torch.bfloat16]


def _model(device):
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(64, 256), nn.GELU(), nn.Linear(256, 16)).to(device)


def _batch(device):
    generator = torch.Generator().manual_seed(1)
    x = torch.randn(32, 64, generator=generator)
    y = torch.randn(32, 16, generator=generator)
    return x.to(device), y.to(device)


def _train(device, dtype, steps=4):
    """Losses, per-step grad sums and final parameters; dtype None is float32."""
    model = _model(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler(_region(device),
                                  enabled=dtype == torch.float16)
    x, y = _batch(device)
    losses, grad_sums = [], []
    for _ in range(steps):
        optimizer.zero_grad()
        with torch.autocast(_region(device), dtype=dtype or torch.float16,
                            enabled=dtype is not None):
            loss = ((model(x) - y) ** 2).mean()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_sums.append(float(sum(p.grad.float().abs().sum()
                                   for p in model.parameters())))
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss))
    params = [p.detach().float().cpu().numpy() for p in model.parameters()]
    return np.array(losses), np.array(grad_sums), params


class TestTorchAmpTrainingLoop(cu.JittorTestCase):

    def test_gradients_of_an_autocast_backward_can_be_read(self, device):
        """The original report, at the default ``fuse_op_limit``."""
        self.assertGreater(jt.flags.fuse_op_limit, 0)
        for dtype in _half_dtypes(device):
            with self.subTest(dtype=str(dtype)):
                model = _model(device)
                x, y = _batch(device)
                with torch.autocast(_region(device), dtype=dtype):
                    loss = ((model(x) - y) ** 2).mean()
                loss.backward()
                total = float(sum(p.grad.float().abs().sum()
                                  for p in model.parameters()))
                self.assertTrue(np.isfinite(total) and total > 0, total)

    def test_a_parameter_grad_keeps_the_parameter_dtype(self, device):
        for dtype in _half_dtypes(device):
            with self.subTest(dtype=str(dtype)):
                model = _model(device)
                x, y = _batch(device)
                with torch.autocast(_region(device), dtype=dtype):
                    loss = ((model(x) - y) ** 2).mean()
                loss.backward()
                for name, p in model.named_parameters():
                    self.assertEqual(str(p.grad.dtype), str(p.dtype), name)

    def test_the_amp_loop_trains_like_the_float32_loop(self, device):
        reference_losses, _, reference_params = _train(device, None)
        for dtype in _half_dtypes(device):
            with self.subTest(dtype=str(dtype)):
                losses, grad_sums, params = _train(device, dtype)
                self.assertTrue(np.isfinite(losses).all(), losses)
                self.assertTrue(np.isfinite(grad_sums).all(), grad_sums)
                # It trains: four AdamW steps on a fixed batch lower the loss.
                self.assertLess(losses[-1], losses[0] * 0.95, losses)
                np.testing.assert_allclose(losses, reference_losses, rtol=1e-2)
                for got, want in zip(params, reference_params):
                    np.testing.assert_allclose(got, want, atol=2e-2)


instantiate_device_type_tests(TestTorchAmpTrainingLoop, globals())


if __name__ == "__main__":
    unittest.main()
