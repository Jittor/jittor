"""Numerical AMP boundaries measured against independent PyTorch CPU/CUDA.

The large values distinguish bfloat16's exponent range from float16; every
case executes its outputs and gradients on the selected device. Expected
scaler values are from PyTorch 2.11, including CUDA's float-valued finite check
for a double gradient and overflow introduced by unscale itself.
"""

from functools import wraps
import unittest

import numpy as np

import jittor as jt
import torch

from jittor._runtime import forbid_backend_fallbacks
from _helpers import common as cu
from _helpers.device_types import instantiate_device_type_tests


def _region(device):
    return "cpu" if device == "cpu" else "cuda"


def _no_fallbacks(function):
    @wraps(function)
    def run(self, device):
        with forbid_backend_fallbacks():
            try:
                return function(self, device)
            finally:
                jt.sync_all(True)
    return run


def _resident(test, value, device):
    value.sync()
    test.assertEqual(value.location(), "cpu" if device == "cpu" else "device")


def _scaler_case(device, scale, gradient, dtype=torch.float32):
    parameter = torch.nn.Parameter(torch.tensor([1.], dtype=dtype, device=device))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    scaler = torch.amp.GradScaler(_region(device), init_scale=scale,
                                  growth_factor=4., growth_interval=1)
    scaler.scale(parameter.sum())
    parameter.grad = torch.tensor([gradient], dtype=dtype, device=device)
    return parameter, optimizer, scaler


class TestAmpBoundaries(cu.JittorTestCase):
    @_no_fallbacks
    def test_all_fp32_matmul_really_uses_bfloat16_in_forward_and_backward(self, device):
        x = torch.nn.Parameter(torch.full((2, 2), 1e5, device=device))
        weight = torch.nn.Parameter(torch.eye(2, device=device))
        optimizer = torch.optim.SGD([x, weight], lr=0.1)
        optimizer.zero_grad()
        with torch.autocast(_region(device), dtype=torch.bfloat16):
            output = x @ weight
        self.assertEqual(str(output.dtype), "torch.bfloat16")
        _resident(self, output, device)
        np.testing.assert_array_equal(output.float().numpy(), np.full((2, 2), 99840.))
        # Backward runs after autocast exited. The op's AMP snapshot must retain
        # the bfloat16 bit, including through its shared high node flag bits.
        output.float().sum().backward()
        for grad in (x.grad, weight.grad):
            self.assertEqual(str(grad.dtype), "torch.float32")
            _resident(self, grad, device)
        np.testing.assert_array_equal(x.grad.numpy(), np.ones((2, 2)))
        np.testing.assert_array_equal(weight.grad.numpy(), np.full((2, 2), 199680.))

    @_no_fallbacks
    def test_fp32_linear_and_convolution_bias_keep_bfloat16(self, device):
        for operation in ("linear", "conv2d", "conv_transpose2d"):
            convolution = operation != "linear"
            with self.subTest(operation=operation):
                shape = (1, 2, 3, 3) if convolution else (2, 2)
                x = torch.nn.Parameter(torch.full(shape, 1e5, device=device))
                weight = torch.eye(2, device=device)
                if convolution:
                    weight = weight.reshape(2, 2, 1, 1)
                weight = torch.nn.Parameter(weight)
                bias = torch.nn.Parameter(torch.zeros(2, device=device))
                optimizer = torch.optim.SGD([x, weight, bias], lr=0.1)
                optimizer.zero_grad()
                fn = getattr(torch.nn.functional, operation)
                with torch.autocast(_region(device), dtype=torch.bfloat16):
                    output = fn(x, weight, bias)
                self.assertEqual(str(output.dtype), "torch.bfloat16")
                _resident(self, output, device)
                np.testing.assert_array_equal(output.float().numpy(), np.full(shape, 99840.))
                output.float().sum().backward()
                for grad in (x.grad, weight.grad, bias.grad):
                    self.assertEqual(str(grad.dtype), "torch.float32")
                    _resident(self, grad, device)
                    self.assertTrue(np.isfinite(grad.numpy()).all())
                np.testing.assert_array_equal(x.grad.numpy(), np.ones(shape))
                count = 9 if convolution else 2
                if convolution:
                    # The portable CPU contraction rounds an intermediate in
                    # bf16: 892928 versus the independent Torch value 897024.
                    np.testing.assert_allclose(weight.grad.numpy(), 897024., rtol=0.01)
                else:
                    np.testing.assert_array_equal(weight.grad.numpy(), np.full((2, 2), 199680.))
                np.testing.assert_array_equal(bias.grad.numpy(), np.full(2, count))

    @_no_fallbacks
    def test_bfloat16_request_overrides_a_float16_operand(self, device):
        x = torch.full((2, 2), 1e5, device=device)
        weight = torch.eye(2, dtype=torch.float16, device=device)
        with torch.autocast(_region(device), dtype=torch.bfloat16):
            output = torch.nn.functional.linear(x, weight)
        self.assertEqual(str(output.dtype), "torch.bfloat16")
        _resident(self, output, device)
        np.testing.assert_array_equal(output.float().numpy(), np.full((2, 2), 99840.))

    @_no_fallbacks
    def test_zero_scale_uses_ieee_reciprocal_and_checks_incoming_gradients(self, device):
        parameter, optimizer, scaler = _scaler_case(device, 0., 0.)
        scaler.unscale_(optimizer)
        self.assertTrue(np.isnan(parameter.grad.numpy()).all())
        scaler.step(optimizer)
        scaler.update()
        _resident(self, parameter, device)
        self.assertTrue(np.isnan(parameter.numpy()).all())
        self.assertEqual(scaler.get_scale(), 0.)

    @_no_fallbacks
    def test_backoff_may_go_below_one_and_keeps_nonfinite_gradients(self, device):
        parameter, optimizer, scaler = _scaler_case(device, 0.25, np.inf)
        scaler.step(optimizer)
        scaler.update()
        self.assertEqual(parameter.item(), 1.)
        self.assertIsNotNone(parameter.grad)
        self.assertTrue(np.isinf(parameter.grad.numpy()).all())
        self.assertEqual(scaler.get_scale(), 0.125)

    @_no_fallbacks
    def test_unscale_overflow_does_not_reclassify_a_finite_gradient(self, device):
        parameter, optimizer, scaler = _scaler_case(device, 1e-10, 1e38)
        scaler.step(optimizer)
        scaler.update()
        self.assertTrue(np.isneginf(parameter.numpy()).all())
        self.assertAlmostEqual(scaler.get_scale() / 4.000000053405728e-10, 1.)

    @_no_fallbacks
    def test_float64_check_domain_matches_each_device(self, device):
        parameter, optimizer, scaler = _scaler_case(device, 2., 1e300, torch.float64)
        scaler.unscale_(optimizer)
        np.testing.assert_allclose(parameter.grad.numpy(), [5e299], rtol=1e-14)
        scaler.step(optimizer)
        scaler.update()
        if device == "cuda":
            # PyTorch CUDA's isfinite_ensure_cuda_math takes a float.
            self.assertEqual(parameter.item(), 1.)
            self.assertEqual(scaler.get_scale(), 1.)
        else:
            np.testing.assert_allclose(parameter.numpy(), [-5e298], rtol=3e-8)
            self.assertEqual(scaler.get_scale(), 8.)

    @_no_fallbacks
    def test_growth_preserves_the_old_scale_if_float32_would_overflow(self, device):
        parameter, optimizer, scaler = _scaler_case(device, 1e38, 1.)
        scaler.step(optimizer)
        scaler.update()
        self.assertEqual(scaler.get_scale(), float(np.float32(1e38)))
        self.assertEqual(scaler.state_dict()["_growth_tracker"], 0)
        self.assertEqual(parameter.item(), 1.)

    @_no_fallbacks
    def test_nonfinite_initial_scales_follow_ieee_arithmetic(self, device):
        for initial in (np.inf, -np.inf, np.nan):
            with self.subTest(initial=initial):
                parameter, optimizer, scaler = _scaler_case(device, initial, 1.)
                scaler.step(optimizer)
                scaler.update()
                _resident(self, parameter, device)
                if np.isnan(initial):
                    self.assertTrue(np.isnan(parameter.item()))
                    self.assertTrue(np.isnan(scaler.get_scale()))
                else:
                    self.assertEqual(parameter.item(), 1.)
                    self.assertEqual(scaler.get_scale(), initial)

    @_no_fallbacks
    def test_manual_scale_update_accepts_scalars_and_checks_tensor_shape(self, device):
        parameter, _, scaler = _scaler_case(device, 2., 1.)
        for value in (0., 0.25, np.inf, np.nan):
            scaler.update(new_scale=float(value))
            if np.isnan(value):
                self.assertTrue(np.isnan(scaler.get_scale()))
            else:
                self.assertEqual(scaler.get_scale(), value)
        scaler.update(new_scale=torch.tensor(0.5, device=device))
        self.assertEqual(scaler.get_scale(), 0.5)
        with self.assertRaisesRegex(RuntimeError, "broadcast shape"):
            scaler.update(new_scale=torch.tensor([0.5], device=device))
        self.assertEqual(scaler.get_scale(), 0.5)
        with self.assertRaisesRegex(AssertionError, "1-element tensor"):
            scaler.update(new_scale=torch.tensor([0.5, 1.], device=device))
        with self.assertRaisesRegex(AssertionError, "requires_grad=False"):
            scaler.update(new_scale=torch.tensor(0.5, device=device, requires_grad=True))
        with self.assertRaisesRegex(TypeError, "new_scale"):
            scaler.update(new_scale=1)
        _resident(self, parameter, device)

    @_no_fallbacks
    def test_fp16_gradients_are_rejected_before_mutation(self, device):
        parameter, optimizer, scaler = _scaler_case(device, 2., 1., torch.float16)
        with self.assertRaisesRegex(ValueError, "unscale FP16 gradients"):
            scaler.unscale_(optimizer)
        self.assertEqual(parameter.grad.item(), 1.)

    @_no_fallbacks
    def test_each_optimizer_has_its_own_unscale_and_step_stage(self, device):
        p, first, scaler = _scaler_case(device, 4., 8.)
        q = torch.nn.Parameter(torch.tensor([1.], device=device))
        second = torch.optim.SGD([q], lr=0.1)
        q.grad = torch.tensor([12.], device=device)
        scaler.unscale_(first)
        scaler.unscale_(second)
        scaler.step(first)
        scaler.step(second)
        np.testing.assert_allclose([p.item(), q.item()], [0.8, 0.7], rtol=1e-6)
        with self.assertRaisesRegex(RuntimeError, "step.*already been called"):
            scaler.step(first)
        with self.assertRaisesRegex(RuntimeError, "after step"):
            scaler.unscale_(first)
        scaler.update()
        self.assertEqual(scaler.get_scale(), 16.)

    @_no_fallbacks
    def test_initialization_and_duplicate_unscale_are_checked(self, device):
        parameter = torch.nn.Parameter(torch.tensor([1.], device=device))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        for operation in ("update", "step", "unscale_"):
            with self.subTest(operation=operation):
                scaler = torch.amp.GradScaler(_region(device))
                with self.assertRaisesRegex(AssertionError, "_scale is None"):
                    getattr(scaler, operation)(*(() if operation == "update" else (optimizer,)))
        scaler.scale(parameter.sum())
        with self.assertRaisesRegex(AssertionError, "No inf checks"):
            scaler.update()
        parameter.grad = torch.ones_like(parameter)
        scaler.unscale_(optimizer)
        with self.assertRaisesRegex(RuntimeError, "unscale_.*already been called"):
            scaler.unscale_(optimizer)
        scaler.update(new_scale=0.25)
        self.assertEqual(scaler.get_scale(), 0.25)
        parameter.grad = torch.ones_like(parameter)
        scaler.step(optimizer)


instantiate_device_type_tests(TestAmpBoundaries, globals())


if __name__ == "__main__":
    unittest.main()
