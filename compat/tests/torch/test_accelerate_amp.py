"""AMP/autograd regressions used by Accelerate's mixed-precision path.

The native fp16 scalar type must be value-initializable because Jittor's fused
reduction kernels initialize their accumulator before applying the reduction.
This is exercised directly here so a regression is caught before the same path
is reached through ``Accelerator.backward``.
"""

import os
import unittest
from contextlib import nullcontext

import numpy as np


try:
    import torch
    import jittor as jt
    from accelerate import Accelerator
    from accelerate.state import AcceleratorState
except ImportError:  # pragma: no cover - optional downstream dependency
    torch = None


_DEVICE = os.environ.get("JITTOR_TEST_DEVICES", "cpu").split(",")[0]


def _values(value):
    return value.detach().clone().cpu().numpy().copy()


@unittest.skipIf(torch is None, "Accelerate is not installed")
class TestAccelerateAmpAutograd(unittest.TestCase):
    def setUp(self):
        AcceleratorState._reset_state(reset_partial_state=True)
        self.addCleanup(AcceleratorState._reset_state, reset_partial_state=True)

    def test_autocast_preserves_pointwise_and_mean_input_dtype(self):
        with jt.flag_scope(use_cuda=int(_DEVICE == "cuda")):
            full = torch.tensor([0.25, -0.125], dtype=torch.float32, device=_DEVICE)
            half = full.half()
            saved_amp = int(jt.flags.amp_reg)
            with torch.autocast(_DEVICE, dtype=torch.float16):
                for result in (full + full, full - full, full * full,
                               full / full, 1.0 / full,
                               full.mean(), torch.mean(full),
                               torch.add(full, full), torch.mul(full, full),
                               torch.multiply(full, full),
                               full.square().mean()):
                    self.assertEqual(result.dtype, torch.float32)
                    self.assertEqual(result.device.type, _DEVICE)
                out = torch.empty_like(full)
                self.assertIs(torch.add(full, full, alpha=2, out=out), out)
                np.testing.assert_array_equal(_values(out), [0.75, -0.375])
                self.assertIs(torch.mul(full, 2, out=out), out)
                np.testing.assert_array_equal(_values(out), [0.5, -0.25])
                self.assertIs(torch.multiply(full, 2, out=out), out)
                np.testing.assert_array_equal(_values(out), [0.5, -0.25])
                np.testing.assert_array_equal(_values(torch.add(2, full)), [2.25, 1.875])
                np.testing.assert_array_equal(_values(torch.mul(2, full)), [0.5, -0.25])
                self.assertEqual(half.square().mean().dtype, torch.float16)
                np.testing.assert_array_equal(_values(full.square().mean()), 0.0390625)
            self.assertEqual(int(jt.flags.amp_reg), saved_amp)
            with jt.flag_scope(amp_reg=2):
                self.assertEqual((full * full).dtype, torch.float16)
            if _DEVICE == "cuda":
                host = torch.tensor([0.25, -0.125], dtype=torch.float32, device="cpu")
                with torch.autocast("cuda", dtype=torch.float16):
                    for result in (host + host, host - host, host * host,
                                   host.mean(), torch.mean(host),
                                   torch.add(host, host), torch.mul(host, host),
                                   torch.multiply(host, host)):
                        self.assertEqual(result.dtype, torch.float32)
                        self.assertEqual(result.device.type, "cpu")

    def test_functional_autocast_preserves_native_and_subclass_protocol(self):
        observed = []

        class CustomParameter(torch.nn.Parameter):
            def __add__(self, other):
                observed.append("add")
                return torch.full_like(self, 101.0)

            def __mul__(self, other):
                observed.append("mul")
                return torch.full_like(self, 202.0)

        with jt.flag_scope(use_cuda=int(_DEVICE == "cuda")):
            custom = CustomParameter(torch.tensor([0.25], device=_DEVICE))
            raw = jt.array([0.25], dtype="float32")
            raw = raw.cuda() if _DEVICE == "cuda" else raw.cpu()
            for inside in (False, True):
                scope = (torch.autocast(_DEVICE, dtype=torch.float16)
                         if inside else nullcontext())
                with scope:
                    for source in (custom, raw):
                        for result, expected in (
                                (torch.add(source, source), [0.5]),
                                (torch.mul(source, source), [0.0625]),
                                (torch.multiply(source, source), [0.0625])):
                            self.assertEqual(result.dtype, torch.float32)
                            self.assertEqual(jt.core.dispatch_context([result])[0], _DEVICE)
                            np.testing.assert_array_equal(_values(result), expected)
                    np.testing.assert_array_equal(
                        _values(torch.add(raw, raw, alpha=2)), [0.75])
            self.assertEqual(observed, [])

    def test_optimizer_step_inside_autocast_preserves_fp32_parameters_and_state(self):
        with jt.flag_scope(use_cuda=int(_DEVICE == "cuda")):
            for kind in ("adamw", "sgd"):
                for inside in (False, True):
                    reference = np.array([0.1, -0.3], dtype=np.float64)
                    momentum = np.zeros(2)
                    variance = np.zeros(2)
                    gradient = np.array([0.025, -0.05], dtype=np.float32)
                    parameter = torch.nn.Parameter(torch.tensor(
                        reference.astype(np.float32), device=_DEVICE))
                    if kind == "adamw":
                        optimizer = torch.optim.AdamW(
                            [parameter], lr=0.03, betas=(0.7, 0.93),
                            eps=1e-4, weight_decay=0.2)
                    else:
                        optimizer = torch.optim.SGD(
                            [parameter], lr=0.03, momentum=0.9, weight_decay=0.2)
                    for step in range(1, 4):
                        optimizer.zero_grad(set_to_none=True)
                        parameter.grad = torch.tensor(gradient, device=_DEVICE)
                        scope = (torch.autocast(_DEVICE, dtype=torch.float16)
                                 if inside else nullcontext())
                        with scope:
                            optimizer.step()
                        if kind == "adamw":
                            reference *= 1.0 - 0.03 * 0.2
                            momentum = 0.7 * momentum + 0.3 * gradient
                            variance = 0.93 * variance + 0.07 * gradient ** 2
                            denominator = (np.sqrt(variance) / np.sqrt(1.0 - 0.93 ** step)
                                           + 1e-4)
                            reference -= momentum * (0.03 / (1.0 - 0.7 ** step)) / denominator
                            expected = {"exp_avg": momentum, "exp_avg_sq": variance}
                            self.assertEqual(float(optimizer.state[parameter]["step"]), step)
                        else:
                            momentum = 0.9 * momentum + gradient + 0.2 * reference
                            reference -= 0.03 * momentum
                            expected = {"momentum_buffer": momentum}
                        self.assertEqual(parameter.dtype, torch.float32)
                        self.assertEqual(parameter.device.type, _DEVICE)
                        np.testing.assert_allclose(_values(parameter), reference,
                                                   atol=1e-6, rtol=1e-6)
                        for key, value in expected.items():
                            state = optimizer.state[parameter][key]
                            self.assertEqual(state.dtype, torch.float32)
                            self.assertEqual(state.device.type, _DEVICE)
                            np.testing.assert_allclose(_values(state), value,
                                                       atol=1e-7, rtol=1e-5)

    @unittest.skipUnless(_DEVICE == "cuda", "CUDA closure autocast contract")
    def test_optimizer_math_scope_preserves_closure_autocast(self):
        with jt.flag_scope(use_cuda=1):
            model = torch.nn.Linear(4, 1).to(device="cuda", dtype=torch.float32)
            with torch.no_grad():
                model.weight.fill_(0.0625)
                model.bias.zero_()
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.03)
            inputs = torch.full((2, 4), 0.125, device="cuda", dtype=torch.float32)
            observed = []

            def closure():
                optimizer.zero_grad(set_to_none=True)
                output = model(inputs)
                observed.append((torch.is_autocast_enabled("cuda"), output.dtype,
                                 int(jt.flags.amp_reg)))
                loss = output.float().square().mean()
                loss.backward()
                return loss

            saved_amp = int(jt.flags.amp_reg)
            with torch.autocast("cuda", dtype=torch.float16):
                optimizer.step(closure)
            self.assertEqual(observed, [(True, torch.float16, 2)])
            self.assertEqual(int(jt.flags.amp_reg), saved_amp)
            for parameter in model.parameters():
                self.assertEqual(parameter.dtype, torch.float32)
                self.assertEqual(parameter.device.type, "cuda")
                self.assertTrue(torch.isfinite(parameter).all().item())
                for key in ("exp_avg", "exp_avg_sq"):
                    state = optimizer.state[parameter][key]
                    self.assertEqual(state.dtype, torch.float32)
                    self.assertEqual(state.device.type, "cuda")

    def test_default_scale_cast_chain_has_finite_scaled_gradient(self):
        with jt.flag_scope(use_cuda=int(_DEVICE == "cuda")):
            parameter = torch.nn.Parameter(torch.tensor(
                [0.25, -0.125], dtype=torch.float32, device=_DEVICE))
            scaler = torch.amp.GradScaler(_DEVICE)
            self.assertEqual(scaler.get_scale(), 65536.0)
            loss = parameter.half().float().square().mean()
            scaler.scale(loss).backward()
            np.testing.assert_array_equal(_values(parameter.grad), [16384.0, -8192.0])

    def test_scaled_overflow_skips_and_backs_off_before_finite_update(self):
        with jt.flag_scope(use_cuda=int(_DEVICE == "cuda")):
            parameter = torch.nn.Parameter(torch.tensor(
                [1.0], dtype=torch.float32, device=_DEVICE))
            optimizer = torch.optim.AdamW([parameter], lr=0.1, weight_decay=0.0)
            scaler = torch.amp.GradScaler(_DEVICE)
            before = _values(parameter)
            scaler.scale((parameter * float("inf")).sum()).backward()
            self.assertTrue(np.isposinf(_values(parameter.grad)).all())
            scaler.step(optimizer)
            scaler.update()
            np.testing.assert_array_equal(_values(parameter), before)
            self.assertEqual(len(optimizer.state), 0)
            self.assertEqual(scaler.get_scale(), 32768.0)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(parameter.square().mean()).backward()
            self.assertTrue(torch.isfinite(parameter.grad).all().item())
            scaler.step(optimizer)
            scaler.update()
            self.assertFalse(np.array_equal(_values(parameter), before))
            self.assertEqual(scaler.get_scale(), 32768.0)

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
        with jt.flag_scope(use_cuda=0):
            accelerator = Accelerator(mixed_precision="fp16", cpu=True)
            model = torch.nn.Linear(4, 2).to(device="cpu")
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            model, optimizer = accelerator.prepare(model, optimizer)
            inputs = torch.randn((2, 4), dtype=torch.float32, device="cpu")
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
            with torch.no_grad():
                model.weight.fill_(0.0625)
                model.bias.zero_()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            model, optimizer = accelerator.prepare(model, optimizer)
            inputs = torch.full((2, 4), 0.125, dtype=torch.float32, device=accelerator.device)
            self.assertEqual(model.weight.device.type, "cuda")
            self.assertEqual(inputs.device.type, "cuda")
            with accelerator.autocast():
                output = model(inputs)
                loss = output.square().mean()
            accelerator.backward(loss)
            # Accelerate wraps prepared model.forward with
            # convert_outputs_to_fp32, matching PyTorch Accelerate's public
            # output contract even though the internal matmul ran in fp16.
            self.assertEqual(str(output.dtype), "torch.float32")
            self.assertEqual(str(loss.dtype), "torch.float32")
            self.assertEqual(accelerator.scaler.get_scale(), 65536.0)
            self.assertTrue(all(p.grad is not None for p in model.parameters()))
            self.assertTrue(all(torch.isfinite(p.grad).all().item() for p in model.parameters()))
            before = model.weight.detach().clone().cpu().numpy().copy()
            self.assertEqual(model.weight.device.type, "cuda")
            optimizer.step()
            self.assertEqual(model.weight.device.type, "cuda")
            after = model.weight.detach().clone().cpu().numpy().copy()
            self.assertTrue((after != before).any())
            if accelerator.scaler is not None:
                self.assertGreater(accelerator.scaler.get_scale(), 0)


if __name__ == "__main__":
    unittest.main()
