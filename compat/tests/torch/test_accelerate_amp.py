"""AMP/autograd regressions used by Accelerate's mixed-precision path.

The native fp16 scalar type must be value-initializable because Jittor's fused
reduction kernels initialize their accumulator before applying the reduction.
This is exercised directly here so a regression is caught before the same path
is reached through ``Accelerator.backward``.
"""

import unittest


try:
    import torch
    import jittor as jt
    from accelerate import Accelerator
except ImportError:  # pragma: no cover - optional downstream dependency
    torch = None


@unittest.skipIf(torch is None, "Accelerate is not installed")
class TestAccelerateAmpAutograd(unittest.TestCase):
    def test_float16_reduction_backward_initializes_native_scalar(self):
        # Jittor exposes gradients on registered parameters (as used by
        # Accelerate optimizers); bare leaf Vars intentionally do not retain a
        # ``.grad`` attribute after backward.
        # The shim defaults to its configured accelerator.  Keep this case a
        # genuine CPU check by scoping native dispatch and placing both inputs
        # explicitly, so a CUDA-enabled environment does not mix placements.
        with jt.flag_scope(use_cuda=0):
            model = torch.nn.Linear(4, 3).to(device="cpu", dtype=torch.float16)
            x = torch.randn((2, 4), dtype=torch.float16, device="cpu")
            loss = model(x).square().mean()
            loss.backward()
            weight = next(model.parameters())
            self.assertEqual(weight.grad.dtype, torch.float16)
            self.assertTrue(torch.isfinite(weight.grad).all().item())

    def test_bfloat16_reduction_backward_initializes_native_scalar(self):
        with jt.flag_scope(use_cuda=0):
            model = torch.nn.Linear(4, 3).to(device="cpu", dtype=torch.bfloat16)
            x = torch.randn((2, 4), dtype=torch.bfloat16, device="cpu")
            loss = model(x).square().mean()
            loss.backward()
            weight = next(model.parameters())
            self.assertEqual(weight.grad.dtype, torch.bfloat16)
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

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_autocast_linear_gradscaler_and_matmul_backward(self):
        # Exercise the real CUDA path used by Accelerate.  In particular, the
        # matmul backward receives an fp16 autocast gradient while the saved
        # Linear operands remain fp32; the native cuBLAS gradient must align
        # that dtype before dispatch.
        with jt.flag_scope(use_cuda=1):
            direct = torch.nn.Linear(4, 3).to(device="cuda", dtype=torch.float32)
            direct_input = torch.randn((2, 4), dtype=torch.float32, device="cuda")
            with torch.autocast("cuda", dtype=torch.float16):
                direct_output = direct(direct_input)
                direct_loss = direct_output.square().mean()
            direct_loss.backward()
            self.assertEqual(str(direct_output.dtype), "torch.float16")
            self.assertTrue(torch.isfinite(direct.weight.grad).all().item())

            bf_model = torch.nn.Linear(4, 3).to(device="cuda", dtype=torch.bfloat16)
            bf_input = torch.randn((2, 4), dtype=torch.bfloat16, device="cuda")
            bf_output = bf_model(bf_input)
            bf_output.square().mean().backward()
            self.assertEqual(str(bf_output.dtype), "torch.bfloat16")
            self.assertEqual(str(bf_model.weight.grad.dtype), "torch.bfloat16")
            self.assertTrue(torch.isfinite(bf_model.weight.grad).all().item())

            accelerator = Accelerator(mixed_precision="fp16")
            model = torch.nn.Linear(4, 3)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            model, optimizer = accelerator.prepare(model, optimizer)
            inputs = torch.randn((2, 4), dtype=torch.float32, device=accelerator.device)
            with accelerator.autocast():
                output = model(inputs)
                loss = output.square().mean()
            accelerator.backward(loss)
            # Accelerate wraps prepared model.forward with
            # convert_outputs_to_fp32, matching PyTorch Accelerate's public
            # output contract even though the internal matmul ran in fp16.
            self.assertEqual(str(output.dtype), "torch.float32")
            self.assertTrue(all(p.grad is not None for p in model.parameters()))
            self.assertTrue(all(torch.isfinite(p.grad).all().item() for p in model.parameters()))
            before = model.weight.numpy().copy()
            optimizer.step()
            self.assertTrue((model.weight.numpy() != before).any())
            if accelerator.scaler is not None:
                self.assertGreater(accelerator.scaler.get_scale(), 0)


if __name__ == "__main__":
    unittest.main()
