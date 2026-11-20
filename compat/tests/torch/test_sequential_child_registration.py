"""Replacing a named Torch Sequential child preserves its registration."""

from collections import OrderedDict

import pytest
import torch

from _helpers.device_types import instantiate_device_type_tests


class TestSequentialChildRegistration:
    def test_named_child_replacement_updates_execution_and_parameters(self, device):
        original = torch.nn.Linear(4, 3)
        replacement = torch.nn.Sequential(OrderedDict([("saved", torch.nn.Linear(4, 3))]))
        container = torch.nn.Sequential(OrderedDict([("classifier", original)]))
        container.classifier = replacement

        assert container[0] is container.classifier is container._modules["classifier"] is replacement
        assert set(container.state_dict()) == {"classifier.saved.weight", "classifier.saved.bias"}
        container.to(device)
        assert replacement.saved.weight.device.type == device
        assert original.weight.device.type == "cpu"

        value = torch.ones((2, 4), device=device)
        actual = container(value)
        assert torch.equal(actual, replacement(value))
        actual.sum().backward()
        assert replacement.saved.weight.grad.device.type == device
        assert original.weight.grad is None

        with pytest.raises(TypeError):
            container.classifier = "not a module"
        assert container[0] is replacement
        container.classifier = None
        assert container.classifier is None and container._modules["classifier"] is None
        assert not list(container.parameters())


instantiate_device_type_tests(TestSequentialChildRegistration, globals(), only_for=("cpu", "cuda"))
