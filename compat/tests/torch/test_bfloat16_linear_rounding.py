"""BF16 Linear adds bias before rounding its FP32 accumulator."""

import numpy as np
import pytest
import torch

from _helpers.device_types import instantiate_device_type_tests


class TestBFloat16LinearRounding:
    def test_biasless_module_inference_keeps_the_default_output_dtype(self, device):
        # The Torch-mode native Module call bypasses the Python functional for
        # this case. Its typed C++ constructor must preserve the default option.
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            layer = torch.nn.Linear(2, 2, bias=False, dtype=dtype, device=device).eval()
            with torch.no_grad():
                layer.weight.copy_(torch.tensor([[1., 2.], [3., 4.]], dtype=dtype, device=device))
                for shape in ((2, 2), (2, 1, 2)):
                    x = torch.tensor([[1., 2.], [3., 4.]], dtype=dtype, device=device).reshape(shape)
                    out = layer(x)
                    assert out.dtype == dtype and out.device.type == device
                    assert not out.requires_grad
                    np.testing.assert_array_equal(out.float().cpu().numpy(),
                                                  np.array([[5., 11.], [11., 25.]]).reshape(shape))
            assert layer.weight.grad is None

    def test_bias_rounds_once_and_all_gradients_remain_bfloat16(self, device):
        # 1.0078125**2 + 1/256 lies just above a BF16 midpoint. Rounding
        # the product first lands exactly on that midpoint and rounds down.
        for shape in ((1,), (1, 1), (2, 1, 1)):
            x = torch.full(shape, 1.0078125, dtype=torch.bfloat16, device=device,
                           requires_grad=True)
            weight = torch.tensor([[1.0078125]], dtype=torch.bfloat16,
                                  device=device, requires_grad=True)
            bias = torch.tensor([0.00390625], dtype=torch.bfloat16,
                                device=device, requires_grad=True)
            out = torch.nn.functional.linear(x, weight, bias)
            split = torch.nn.functional.linear(x, weight, None) + bias
            assert out.dtype == torch.bfloat16 and out.device.type == device
            np.testing.assert_array_equal(out.float().detach().cpu().numpy(),
                                          np.full(shape, 1.0234375))
            np.testing.assert_array_equal(split.float().detach().cpu().numpy(),
                                          np.full(shape, 1.015625))
            out.float().square().sum().backward()
            for value in (x.grad, weight.grad, bias.grad):
                assert value is not None and value.dtype == torch.bfloat16
                assert value.device.type == device
            np.testing.assert_array_equal(x.grad.float().cpu().numpy(),
                                          np.full(shape, 2.0625))
            np.testing.assert_array_equal(weight.grad.float().cpu().numpy(),
                                          [[2.0625 * x.numel()]])
            np.testing.assert_array_equal(bias.grad.float().cpu().numpy(),
                                          [2.046875 * x.numel()])

    def test_biasless_module_training_keeps_all_native_gradients(self, device):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            layer = torch.nn.Linear(2, 2, bias=False, dtype=dtype, device=device)
            with torch.no_grad():
                layer.weight.copy_(torch.tensor([[1., 2.], [3., 4.]], dtype=dtype, device=device))
            x = torch.tensor([[1., 2.], [3., 4.]], dtype=dtype, device=device, requires_grad=True)
            out = layer(x)
            assert out.dtype == dtype and out.device.type == device
            out.float().sum().backward()
            for value in (x.grad, layer.weight.grad):
                assert value is not None and value.dtype == dtype
                assert value.device.type == device
                np.testing.assert_array_equal(value.float().cpu().numpy(), [[4., 6.], [4., 6.]])

    def test_module_and_autocast_share_the_bias_accumulator(self, device):
        layer = torch.nn.Linear(1, 1, device=device)
        with torch.no_grad():
            layer.weight.fill_(1.0078125)
            # Autocast must round the FP32 bias to its input dtype first.
            layer.bias.fill_(0.00390625 + 0.000001)
        x = torch.tensor([[1.0078125]], device=device, requires_grad=True)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            out = layer(x)
        assert out.dtype == torch.bfloat16 and out.device.type == device
        np.testing.assert_array_equal(out.float().detach().cpu().numpy(), [[1.0234375]])
        out.float().square().sum().backward()
        for value in (x.grad, layer.weight.grad, layer.bias.grad):
            assert value is not None and value.dtype == torch.float32
            assert value.device.type == device
        np.testing.assert_array_equal(x.grad.cpu().numpy(), [[2.0625]])
        np.testing.assert_array_equal(layer.weight.grad.cpu().numpy(), [[2.0625]])
        np.testing.assert_array_equal(layer.bias.grad.cpu().numpy(), [2.046875])

    def test_bad_shapes_raise_before_launch(self, device):
        x = torch.ones((2, 3), dtype=torch.bfloat16, device=device)
        weight = torch.ones((4, 3), dtype=torch.bfloat16, device=device)
        with pytest.raises(RuntimeError):
            torch.nn.functional.linear(x, weight, torch.ones((5,), dtype=torch.bfloat16, device=device))
        with pytest.raises(RuntimeError):
            torch.nn.functional.linear(x, weight[:, :2], torch.ones((4,), dtype=torch.bfloat16, device=device))

    def test_create_graph_keeps_the_native_backward_differentiable(self, device):
        x = torch.tensor([[1.0078125]], dtype=torch.bfloat16, device=device, requires_grad=True)
        weight = torch.tensor([[1.0078125]], dtype=torch.bfloat16, device=device, requires_grad=True)
        bias = torch.tensor([0.00390625], dtype=torch.bfloat16, device=device, requires_grad=True)
        out = torch.nn.functional.linear(x, weight, bias)
        dx = torch.autograd.grad(out.float().square().sum(), x, create_graph=True)[0]
        second = torch.autograd.grad(dx.float().sum(), weight)[0]
        assert dx.dtype == second.dtype == torch.bfloat16
        assert second.device.type == device
        np.testing.assert_array_equal(second.float().cpu().numpy(), [[4.0625]])


instantiate_device_type_tests(TestBFloat16LinearRounding, globals(), only_for=("cpu", "cuda"))
