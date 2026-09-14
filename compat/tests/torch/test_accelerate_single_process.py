"""Focused Accelerate 1.10 single-process contracts for the Torch shim.

These tests intentionally exercise the real DataLoader/optimizer wrappers used
by Accelerate instead of testing the wrapper classes in isolation.
"""

import unittest
from collections import namedtuple


try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    from accelerate import Accelerator
    from accelerate.utils import send_to_device
except ImportError:  # pragma: no cover - optional downstream dependency
    torch = None


@unittest.skipIf(torch is None, "Accelerate is not installed")
class TestAccelerateSingleProcess(unittest.TestCase):
    def test_prepare_training_loop_places_batches_and_updates_scheduler(self):
        if not torch.cuda.is_available():
            self.skipTest("requires a real CUDA device")

        torch.manual_seed(17)
        model = nn.Linear(4, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
        loader = DataLoader(
            TensorDataset(torch.randn(4, 4), torch.randn(4, 2)), batch_size=2
        )
        accelerator = Accelerator(gradient_accumulation_steps=2)
        model, optimizer, loader, scheduler = accelerator.prepare(
            model, optimizer, loader, scheduler
        )

        self.assertEqual(next(model.parameters()).device.type, "cuda")
        self.assertEqual(len(loader), 2)
        initial_lr = scheduler.get_last_lr()[0]
        for inputs, targets in loader:
            self.assertEqual(inputs.device.type, "cuda")
            self.assertEqual(targets.device.type, "cuda")
            with accelerator.accumulate(model):
                loss = ((model(inputs) - targets) ** 2).mean()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        self.assertLess(scheduler.get_last_lr()[0], initial_lr)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_send_to_device_preserves_nested_container_types(self):
        if not torch.cuda.is_available():
            self.skipTest("requires a real CUDA device")

        Pair = namedtuple("Pair", "left right")
        value = {"items": [torch.ones(2), (torch.zeros(2),)], "pair": Pair(torch.ones(1), 3)}
        moved = send_to_device(value, torch.device("cuda"))

        self.assertIsInstance(moved, dict)
        self.assertIsInstance(moved["items"], list)
        self.assertIsInstance(moved["items"][1], tuple)
        self.assertIsInstance(moved["pair"], Pair)
        self.assertEqual(moved["items"][0].device.type, "cuda")
        self.assertEqual(moved["pair"].left.device.type, "cuda")
        self.assertEqual(moved["pair"].right, 3)


if __name__ == "__main__":
    unittest.main()
