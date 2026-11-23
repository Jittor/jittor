import pytest
import torch

from torch.nn.modules.linear import LazyLinear
from torch.nn.parameter import Parameter, UninitializedParameter, is_lazy

from _helpers.device_types import instantiate_device_type_tests


class TestLazyLinear:
    def test_optimizer_keeps_parameters_from_before_first_forward(self, device):
        layer = LazyLinear(3).to(device)
        assert LazyLinear is torch.nn.LazyLinear
        weight, bias = tuple(layer.parameters())
        assert isinstance(weight, UninitializedParameter) and is_lazy(bias)
        with pytest.raises(RuntimeError, match="shape.*uninitialized"):
            weight.shape
        optimizer = torch.optim.Adam(layer.parameters(), lr=0.01)
        value = torch.ones((2, 4), device=device)
        output = layer(input=value)
        assert type(layer) is torch.nn.Linear
        assert layer.in_features == 4 and layer.out_features == 3
        assert tuple(output.shape) == (2, 3) and output.device.type == device
        assert type(weight) is Parameter and not is_lazy(bias)
        assert layer.weight is weight and layer.bias is bias
        assert optimizer.param_groups[0]["params"][0] is weight
        previous = weight.detach().clone()
        output.sum().backward()
        assert weight.grad is not None and bias.grad is not None
        optimizer.step()
        assert not torch.equal(weight, previous)

    def test_nested_checkpoint_materializes_without_replacing_weight(self, device):
        eager = torch.nn.Sequential(torch.nn.Linear(4, 2, bias=False, device=device))
        layer = LazyLinear(2, bias=False, device=device)
        container = torch.nn.Sequential(layer)
        weight = layer.weight
        original = container.state_dict(keep_vars=False)
        assert list(original) == ["0.weight"] and is_lazy(original["0.weight"])
        assert not container.load_state_dict(original).missing_keys
        assert is_lazy(weight)
        with torch.no_grad():
            eager[0].weight.fill_(0.125)
        state = eager.state_dict()
        incompatible = container.load_state_dict(state)
        assert not incompatible.missing_keys and not incompatible.unexpected_keys
        assert type(layer) is LazyLinear and layer.in_features == 0
        assert type(weight) is Parameter and layer.weight is weight and layer.bias is None
        assert torch.equal(weight, state["0.weight"])
        with pytest.raises(AssertionError, match="features"):
            container(torch.ones((1, 3), device=device))
        value = torch.ones((2, 4), device=device)
        assert torch.equal(container(value), eager(value))
        assert type(layer) is torch.nn.Linear and layer.in_features == 4

    def test_keeps_dtype_and_requires_grad_after_materialization(self, device):
        layer = LazyLinear(2, device=device).to(dtype=torch.float64)
        layer.weight.requires_grad_(False)
        assert is_lazy(layer.weight) and layer.weight.dtype == torch.float64
        value = torch.ones((2, 4), device=device, dtype=torch.float64)
        output = layer(value)
        assert output.dtype == torch.float64 and tuple(output.shape) == (2, 2)
        assert not layer.weight.requires_grad and layer.bias.requires_grad
        assert output.device.type == device


instantiate_device_type_tests(TestLazyLinear, globals(), only_for=("cpu", "cuda"))
