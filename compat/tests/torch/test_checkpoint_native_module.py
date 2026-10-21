"""Torch checkpoint dispatch accepts native modules without duplicating loaders."""

import numpy as np
import pytest
import torch
import jittor as jt
from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict

from _helpers.device_types import instantiate_device_type_tests


class _NativeModel(jt.Module):
    def __init__(self):
        self.weight = jt.array([1., 2.])
        self.bias = jt.array([3.])

    def execute(self, value):
        return (value * self.weight).sum() + self.bias


class TestCheckpointNativeModule:
    def test_copying_load_preserves_native_parameters_and_aliases(self, device):
        model = _NativeModel()
        weight = model.weight
        alias = weight.detach()
        pointer = weight.device_raw_ptr
        snapshot = jt.Var.copy(weight)
        result = set_model_state_dict(model, {
            "weight": torch.tensor([4., 5.], device=device),
            "bias": torch.tensor([6.], device=device),
        })
        assert not result.missing_keys and not result.unexpected_keys
        assert model.weight is weight
        assert model.weight.device_raw_ptr == pointer
        assert not model.weight.is_stop_grad()
        np.testing.assert_array_equal(snapshot.numpy(), [1., 2.])
        np.testing.assert_array_equal(alias.numpy(), [4., 5.])
        np.testing.assert_array_equal(model.bias.numpy(), [6.])
        gradients = jt.grad(model(jt.array([2., 3.])).sum(), [model.weight, model.bias])
        np.testing.assert_array_equal(gradients[0].numpy(), [2., 3.])
        np.testing.assert_array_equal(gradients[1].numpy(), [1.])

    def test_strict_and_shape_failures_precede_mutation(self, device):
        model = _NativeModel()
        for state, message, strict in (
            ({"weight": torch.tensor([8., 9.], device=device)}, "Missing key", True),
            ({"weight": torch.tensor([8., 9.], device=device),
              "bias": torch.tensor([6.], device=device),
              "extra": torch.tensor([7.], device=device)}, "Unexpected key", True),
            ({"weight": torch.tensor([8.], device=device),
              "bias": torch.tensor([6.], device=device)}, "size mismatch", False),
        ):
            with pytest.raises(RuntimeError, match=message):
                set_model_state_dict(model, state, options=StateDictOptions(strict=strict))
            np.testing.assert_array_equal(model.weight.numpy(), [1., 2.])
            np.testing.assert_array_equal(model.bias.numpy(), [3.])
        result = set_model_state_dict(model, {
            "weight": torch.tensor([8., 9.], device=device),
            "extra": torch.tensor([7.], device=device),
        }, options=StateDictOptions(strict=False))
        assert result.missing_keys == ["bias"]
        assert result.unexpected_keys == ["extra"]
        np.testing.assert_array_equal(model.weight.numpy(), [8., 9.])
        np.testing.assert_array_equal(model.bias.numpy(), [3.])

    def test_custom_native_loader_override_remains_authoritative(self, device):
        class CustomModel(_NativeModel):
            def load_state_dict(self, state_dict, strict=True):
                self.calls = getattr(self, "calls", 0) + 1
                self.strict_received = strict
                return torch.nn.Module.load_state_dict(self, state_dict, strict=strict)

        model = CustomModel()
        result = set_model_state_dict(model, {"weight": torch.tensor([8., 9.], device=device)},
                                      options=StateDictOptions(strict=False))
        assert model.calls == 1 and model.strict_received is False
        assert result.missing_keys == ["bias"]
        np.testing.assert_array_equal(model.weight.numpy(), [8., 9.])


instantiate_device_type_tests(TestCheckpointNativeModule, globals(), only_for=("cpu", "cuda"))
