"""Sharded AMP math on each selected device; local collective plumbing is mocked.

The companion NCCL test checks real cross-rank overflow synchronization.
"""
from contextlib import contextmanager
from unittest import mock

import numpy as np
import torch
import jittor as jt
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

from _helpers import common as cu
from _helpers.device_types import instantiate_device_type_tests
from jittor._runtime import forbid_backend_fallbacks


@contextmanager
def _local_collective(device, remote_overflow=False):
    calls = []

    class Work:
        def __init__(self, value):
            self.value = value

        def wait(self):
            if remote_overflow:
                self.value.fill_(1.0)

    def all_reduce(value, op=None, group=None, async_op=False):
        value.sync()
        assert value.location() == ("cpu" if device == "cpu" else "device")
        assert async_op
        calls.append(value)
        return Work(value)

    with mock.patch.object(torch.distributed, "is_initialized", return_value=True), \
            mock.patch.object(torch.distributed, "get_backend",
                              return_value="gloo" if device == "cpu" else "nccl"), \
            mock.patch.object(torch.distributed, "all_reduce", side_effect=all_reduce):
        yield calls


class TestShardedGradScaler(cu.JittorTestCase):
    def test_fp16_three_real_updates_and_state_restore(self, device):
        with forbid_backend_fallbacks(), _local_collective(device) as calls:
            p = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float16, device=device))
            opt = torch.optim.SGD([p], lr=0.25)
            scaler = ShardedGradScaler(device, init_scale=8., growth_interval=2)
            for step in range(3):
                opt.zero_grad()
                loss = p.float().sum()
                scaled = scaler.scale(loss)
                np.testing.assert_array_equal(scaled.numpy(), loss.numpy() * scaler.get_scale())
                scaled.backward()
                scaler.unscale_(opt)
                np.testing.assert_array_equal(p.grad.numpy(), [1.0])
                scaler.step(opt)
                scaler.update()
                p.sync()
                self.assertEqual(p.location(), "cpu" if device == "cpu" else "device")
                np.testing.assert_array_equal(p.numpy(), [1.0 - (step + 1) * 0.25])
                if step == 0:
                    saved = scaler.state_dict()
                    restored = ShardedGradScaler(device, init_scale=1.)
                    restored.load_state_dict(saved)
                    self.assertEqual(restored.state_dict(), saved)
                    scaler = restored
            self.assertEqual(len(calls), 3)
            self.assertEqual(scaler.get_scale(), 16.)
            self.assertEqual(set(scaler.state_dict()), {
                "scale", "growth_factor", "backoff_factor", "growth_interval", "_growth_tracker"})

    def test_nonfinite_skips_step_and_backs_off(self, device):
        for bad in (float("inf"), float("nan")):
            with self.subTest(bad=bad), forbid_backend_fallbacks(), _local_collective(device):
                p = torch.nn.Parameter(torch.tensor([1.], dtype=torch.float16, device=device))
                opt = torch.optim.SGD([p], lr=0.25)
                scaler = ShardedGradScaler(device, init_scale=8.)
                opt.zero_grad()
                scaler.scale(p.float().sum()).backward()
                p.grad.fill_(bad)
                with mock.patch.object(opt, "step", wraps=opt.step) as step:
                    self.assertIsNone(scaler.step(opt))
                    step.assert_not_called()
                scaler.update()
                np.testing.assert_array_equal(p.numpy(), [1.])
                self.assertEqual(scaler.get_scale(), 4.)

    def test_empty_rank_still_collects_remote_overflow(self, device):
        class EmptyOptimizer:
            param_groups = [{"params": [], "grads": []}]

            def step(self):
                raise AssertionError("remote overflow must skip empty-rank optimizer")

        with forbid_backend_fallbacks(), _local_collective(device, remote_overflow=True) as calls:
            scaler = ShardedGradScaler(device, init_scale=8.)
            scaler.scale(torch.tensor(1., device=device))
            self.assertIsNone(scaler.step(EmptyOptimizer()))
            scaler.update()
            self.assertEqual(len(calls), 1)
            self.assertEqual(scaler.get_scale(), 4.)

    def test_regular_scaler_still_rejects_fp16(self, device):
        p = torch.nn.Parameter(torch.tensor([1.], dtype=torch.float16, device=device))
        opt = torch.optim.SGD([p], lr=0.25)
        scaler = torch.amp.GradScaler(device, init_scale=8.)
        opt.zero_grad()
        scaler.scale(p.float().sum()).backward()
        with self.assertRaisesRegex(ValueError, "FP16"):
            scaler.unscale_(opt)

    def test_duplicate_unscale_is_rejected(self, device):
        with forbid_backend_fallbacks(), _local_collective(device):
            p = torch.nn.Parameter(torch.tensor([1.], dtype=torch.float16, device=device))
            opt = torch.optim.SGD([p], lr=0.25)
            scaler = ShardedGradScaler(device, init_scale=8.)
            opt.zero_grad()
            scaler.scale(p.float().sum()).backward()
            scaler.unscale_(opt)
            with self.assertRaisesRegex(RuntimeError, "already been called"):
                scaler.unscale_(opt)

    def test_captured_scale_cannot_reuse_device_state_in_eager_unscale(self, device):
        from jittor._runtime import step_capture

        with forbid_backend_fallbacks(), _local_collective(device) as calls:
            scaler = ShardedGradScaler(device, init_scale=8., growth_interval=1)
            # Use the real capture entry, not a mocked active flag or a fake
            # _device slot. The second call runs scale() while capture is live
            # and initializes the native device scale/tracker tensors.
            run = jt.capture_step(lambda value: scaler.scale(value), record=False)
            value = torch.tensor(1., device=device)
            run(value).sync()
            run(value).sync()
            self.assertIsNotNone(scaler._device)
            self.assertFalse(step_capture.active())

            p = torch.nn.Parameter(torch.tensor([1.], dtype=torch.float16, device=device))
            opt = torch.optim.SGD([p], lr=0.25)
            p.grad = torch.tensor([8.], dtype=torch.float16, device=device)
            with self.assertRaisesRegex(NotImplementedError, "fresh scaler"):
                scaler.unscale_(opt)
            # Refusal must precede all gradient mutation and collectives.
            np.testing.assert_array_equal(p.grad.numpy(), [8.])
            self.assertEqual(calls, [])

            restored = ShardedGradScaler(device, init_scale=8., growth_interval=1)
            opt.zero_grad()
            restored.scale(p.float().sum()).backward()
            restored.step(opt)
            restored.update()
            np.testing.assert_array_equal(p.numpy(), [.75])
            self.assertEqual(restored.get_scale(), 16.)
            self.assertEqual(len(calls), 1)

    def test_disabled_scaler_needs_no_collective(self, device):
        scaler = ShardedGradScaler(device, enabled=False)
        value = torch.tensor(1., device=device)
        self.assertIs(scaler.scale(value), value)
        self.assertFalse(scaler.is_enabled())
        self.assertEqual(scaler.state_dict(), {})
        self.assertEqual(scaler.get_scale(), 1.)


instantiate_device_type_tests(TestShardedGradScaler, globals())
