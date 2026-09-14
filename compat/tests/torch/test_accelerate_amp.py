"""AMP/autograd regressions used by Accelerate's mixed-precision path.

The native fp16 scalar type must be value-initializable because Jittor's fused
reduction kernels initialize their accumulator before applying the reduction.
This is exercised directly here so a regression is caught before the same path
is reached through ``Accelerator.backward``.
"""

import unittest


try:
    import torch
    from accelerate import Accelerator
except ImportError:  # pragma: no cover - optional downstream dependency
    torch = None


@unittest.skipIf(torch is None, "Accelerate is not installed")
class TestAccelerateAmpAutograd(unittest.TestCase):
    def test_float16_reduction_backward_initializes_native_scalar(self):
        # Jittor exposes gradients on registered parameters (as used by
        # Accelerate optimizers); bare leaf Vars intentionally do not retain a
        # ``.grad`` attribute after backward.
        model = torch.nn.Linear(4, 3).to(dtype=torch.float16)
        x = torch.randn((2, 4), dtype=torch.float16)
        loss = model(x).square().mean()
        loss.backward()
        weight = next(model.parameters())
        self.assertEqual(weight.grad.dtype, torch.float16)
        self.assertTrue(torch.isfinite(weight.grad).all().item())

    def test_accelerator_autocast_context_and_grad_scaler_api(self):
        accelerator = Accelerator(mixed_precision="fp16", cpu=True)
        model = torch.nn.Linear(4, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        model, optimizer = accelerator.prepare(model, optimizer)
        inputs = torch.randn((2, 4), dtype=torch.float32)
        with accelerator.autocast():
            loss = model(inputs).square().mean()
        accelerator.backward(loss)
        self.assertTrue(all(p.grad is not None for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()
