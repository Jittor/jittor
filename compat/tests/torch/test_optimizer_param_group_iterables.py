"""Torch module parameters can be appended without manually making a list."""

import numpy as np
import pytest
import torch

from _helpers.common import get_all_device_types
from _optimizer_contract_support import require_optimizer_backend as require_optimizer_backend


def trajectory(name, form, device):
    first = torch.nn.Parameter(torch.tensor([1.0], device=device))
    module = torch.nn.ParameterList([
        torch.nn.Parameter(torch.tensor([2.0, -3.0], device=device)),
        torch.nn.Parameter(torch.tensor([0.5, 4.0, -2.0], device=device)),
    ])
    parameters = list(module.parameters())
    optimizer = getattr(torch.optim, name)([first], lr=0.1)
    (first * first).sum().backward()
    optimizer.step()
    supplied = {"list": parameters, "tuple": tuple(parameters),
                "iterator": iter(parameters), "generator": module.parameters()}[form]
    optimizer.add_param_group({"params": supplied, "lr": 0.03})
    stored = optimizer.param_groups[-1]["params"]
    assert len(stored) == len(parameters)
    assert all(actual is expected for actual, expected in zip(stored, parameters))
    snapshots = []
    for index in range(3):
        optimizer.zero_grad()
        sum((p * p).sum() for p in [first] + parameters).backward()
        optimizer.step()
        step = []
        for parameter in [first] + parameters:
            assert parameter.device.type == device
            if hasattr(torch, "_torch_compat_install_context"):
                parameter.sync()
                assert parameter.location() == ("cpu" if device == "cpu" else "device")
            step.append(parameter.detach().cpu().numpy().copy())
        snapshots.append(step)
        if name == "SGD":
            for value, initial in zip(step[1:], ([2.0, -3.0], [0.5, 4.0, -2.0])):
                np.testing.assert_allclose(
                    value, np.asarray(initial) * 0.94 ** (index + 1), rtol=1e-6)
    for final, initial in zip(snapshots[-1][1:], ([2.0, -3.0], [0.5, 4.0, -2.0])):
        assert not np.array_equal(final, initial), "appended parameter did not update"
    return snapshots


@pytest.mark.parametrize("name", ["SGD", "Adam", "AdamW", "RMSprop", "Adan"])
@pytest.mark.parametrize("form", ["tuple", "iterator", "generator"])
@pytest.mark.parametrize("device", get_all_device_types())
def test_appended_module_parameters_match_list_training(name, form, device):
    actual = trajectory(name, form, device)
    expected = trajectory(name, "list", device)
    for actual_step, expected_step in zip(actual, expected):
        for value, reference in zip(actual_step, expected_step):
            np.testing.assert_allclose(value, reference, rtol=1e-6, atol=1e-7)
