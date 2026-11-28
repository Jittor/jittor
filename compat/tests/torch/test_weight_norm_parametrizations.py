"""Checkpoint and gradient contracts for parametrized weight normalization."""

import numpy as np
import pytest
import torch

from _helpers.device_types import instantiate_device_type_tests


class TestWeightNormParametrizations:
    def test_load_legacy_checkpoint_and_move_originals(self, device):
        linear = torch.nn.Linear(3, 2, bias=False)
        model = torch.nn.Sequential(linear)
        original_class = type(linear)
        gain = torch.tensor([[10.0], [26.0]], device="cpu")
        direction = torch.tensor([[3.0, 4.0, 0.0], [0.0, 5.0, 12.0]], device="cpu")

        assert torch.nn.utils.parametrizations.weight_norm(linear) is linear
        keys = {"0.parametrizations.weight.original0", "0.parametrizations.weight.original1"}
        assert set(model.state_dict()) == keys
        assert set(dict(model.named_parameters())) == keys
        model.load_state_dict({"0.weight_g": gain, "0.weight_v": direction}, strict=True)
        original_parameters = tuple(model.parameters())
        model.to(device=device, dtype=torch.float64)
        for old, current in zip(original_parameters, model.parameters()):
            assert old is current and current.is_leaf and current._base is None
            assert current.device.type == torch.device(device).type and current.dtype == torch.float64

        input = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64, device=device, requires_grad=True)
        output = model(input)
        assert output.device == input.device
        torch.testing.assert_close(output, input.new_tensor([[22.0, 92.0]]))
        output.sum().backward()
        torch.testing.assert_close(input.grad, input.new_tensor([[6.0, 18.0, 24.0]]))
        assert all(p.grad is not None for p in model.parameters())

        parametrize = torch.nn.utils.parametrize
        assert parametrize.is_parametrized(linear, "weight")
        assert parametrize.type_before_parametrizations(linear) is original_class
        with pytest.raises(ValueError, match="leave_parametrized"):
            parametrize.remove_parametrizations(linear, "weight", leave_parametrized=False)
        parametrize.remove_parametrizations(linear, "weight")
        assert set(model.state_dict()) == {"0.weight"}
        assert type(linear) is original_class
        assert not parametrize.is_parametrized(linear)
        torch.testing.assert_close(model(input.detach()), output.detach())

    def test_weight_reads_current_originals_and_preserves_freezing(self, device):
        layer = torch.nn.Linear(2, 1, bias=False, device=device).requires_grad_(False)
        torch.nn.utils.parametrizations.weight_norm(layer, dim=None)
        originals = layer.parametrizations.weight
        assert tuple(originals.original0.shape) == ()
        assert all(p.is_leaf and not p.requires_grad for p in layer.parameters())
        with torch.no_grad():
            layer.weight = torch.tensor([[4.0, 3.0]], device=device)
            originals.original0.copy_(torch.tensor(10.0, device=device))
        torch.testing.assert_close(layer.weight, torch.tensor([[8.0, 6.0]], device=device))

    def test_fp32_inference_weight_rounding(self, device):
        rng = np.random.default_rng(23)
        direction = (rng.integers(-128, 129, size=(4, 37, 9)).astype(np.float32) / 32)
        gain = np.array([0.7, -1.1, 1.9, 3.3], dtype=np.float32).reshape(4, 1, 1)
        layer = torch.nn.Conv1d(37, 4, 9, bias=False)
        torch.nn.utils.parametrizations.weight_norm(layer)
        layer.to(device)
        originals = layer.parametrizations.weight
        with torch.no_grad():
            originals.original0.copy_(torch.from_numpy(gain).to(device))
            originals.original1.copy_(torch.from_numpy(direction).to(device))
            actual = layer.weight
            assert actual.device.type == torch.device(device).type
            actual = actual.cpu().numpy()

        # These dyadic squares and their sums fit exactly in FP32, so the
        # reference is independent of reduction order.
        norm = np.sqrt((direction * direction).sum(axis=(1, 2), keepdims=True))
        if torch.device(device).type == "cuda":
            expected = (gain * direction) * (np.float32(1) / norm)
            np.testing.assert_array_equal(actual, expected)
        else:
            np.testing.assert_allclose(actual, direction * (gain / norm), rtol=2e-7, atol=0)


instantiate_device_type_tests(TestWeightNormParametrizations, globals())
