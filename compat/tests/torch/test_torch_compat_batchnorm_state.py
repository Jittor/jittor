"""BatchNorm's public Torch counter stays scalar, int64 and persistent."""

import io
import os
import unittest

import numpy as np
import torch


class BatchNormState(unittest.TestCase):
    def devices(self):
        for device in os.environ.get("JITTOR_TEST_DEVICES", "cpu").split(","):
            self.assertIn(device, ("cpu", "cuda"))
            if device == "cuda":
                self.assertTrue(torch.cuda.is_available(), "selected CUDA must actually execute")
            yield device

    def assert_counter(self, model, value, device):
        counter = model.num_batches_tracked
        self.assertEqual(tuple(counter.shape), ())
        self.assertEqual(counter.dtype, torch.int64)
        self.assertEqual(counter.detach().cpu().numpy().shape, ())
        self.assertEqual(counter.item(), value)
        self.assertEqual(counter.device.type, device)
        self.assertFalse(counter.requires_grad)
        self.assertIn("num_batches_tracked", dict(model.named_buffers()))
        self.assertNotIn("num_batches_tracked", dict(model.named_parameters()))
        self.assertIn("num_batches_tracked", model.state_dict())

    def test_counter_updates_and_checkpoint_roundtrip_on_actual_device(self):
        shapes = (
            ("BatchNorm1d", (2, 3)),
            ("BatchNorm2d", (2, 3, 4, 4)),
            ("BatchNorm3d", (2, 3, 2, 2, 2)),
        )
        for device in self.devices():
            for layer, shape in shapes:
                with self.subTest(device=device, layer=layer):
                    model = getattr(torch.nn, layer)(3).to(device)
                    self.assert_counter(model, 0, device)
                    for step in range(3):
                        values = np.random.RandomState(step).randn(*shape).astype("float32")
                        output = model(torch.from_numpy(values).to(device))
                        self.assertEqual(output.device.type, device)
                        self.assert_counter(model, step + 1, device)
                    model.eval()
                    model(torch.from_numpy(values).to(device))
                    self.assert_counter(model, 3, device)
                    checkpoint = io.BytesIO()
                    torch.save(model.state_dict(), checkpoint)
                    checkpoint.seek(0)
                    restored = getattr(torch.nn, layer)(3).to(device)
                    restored.load_state_dict(
                        torch.load(checkpoint, weights_only=False), strict=True
                    )
                    self.assert_counter(restored, 3, device)
                    restored.train()
                    restored(torch.from_numpy(values).to(device))
                    self.assert_counter(restored, 4, device)
                    restored.to(dtype=torch.float64)
                    self.assert_counter(restored, 4, device)

    def test_counter_is_none_when_running_statistics_are_disabled(self):
        for device in self.devices():
            model = torch.nn.BatchNorm2d(3, track_running_stats=False).to(device)
            self.assertIsNone(model.num_batches_tracked)
            self.assertNotIn("num_batches_tracked", model.state_dict())
            model(torch.ones((2, 3, 4, 4), device=device))
            self.assertIsNone(model.num_batches_tracked)

    def test_frontend_counter_does_not_change_native_checkpoint_format(self):
        import jittor as jt

        native = jt.nn.BatchNorm(3)
        frontend = torch.nn.BatchNorm2d(3)
        self.assertNotIn("num_batches_tracked", native.state_dict())
        self.assertEqual(tuple(native.num_batches_tracked.shape), (1,))
        self.assertEqual(str(native.num_batches_tracked.dtype), "int32")
        self.assertIn("num_batches_tracked", frontend.state_dict())
        self.assertIsNot(type(native), type(frontend))
