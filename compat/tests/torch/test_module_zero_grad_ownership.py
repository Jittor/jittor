"""Module.zero_grad clears only its parameters, including optimizer slots."""
import os

import numpy as np
import pytest
import torch


@pytest.fixture(params=os.environ.get("JITTOR_TEST_DEVICES", "cpu").split(","))
def device(request):
    return request.param


def _model(device):
    model = torch.nn.Linear(2, 1, bias=False, device=device)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1., 2.]], device=device))
    return model


def _array(value):
    return value.detach().cpu().numpy().copy()


def test_clear_does_not_touch_unrelated_optimizer(device):
    first, second = _model(device), _model(device)
    optimizer = torch.optim.AdamW(second.parameters(), lr=.01)
    first.weight.grad = torch.ones_like(first.weight)
    second.weight.grad = torch.full_like(second.weight, 3.)
    retained = second.weight.grad
    first.zero_grad()
    assert first.weight.grad is None
    assert second.weight.grad is retained
    np.testing.assert_array_equal(_array(retained), [[3., 3.]])
    optimizer.step()
    assert not np.array_equal(_array(second.weight), [[1., 2.]])


def test_zero_keeps_gradient_object_and_accumulation_with_optimizer(device):
    model = _model(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    x = torch.tensor([[2., 3.]], device=device)
    model(x).sum().backward()
    retained = model.weight.grad
    alias = retained.detach()
    pointer = retained.data_ptr()
    model.zero_grad(set_to_none=False)
    assert model.weight.grad is retained
    assert retained.data_ptr() == pointer == alias.data_ptr()
    np.testing.assert_array_equal(_array(alias), [[0., 0.]])
    model(x).sum().backward()
    assert model.weight.grad is retained
    np.testing.assert_array_equal(_array(model.weight.grad), [[2., 3.]])
    optimizer.step()
    assert int(optimizer.state[model.weight]["step"].item()) == 1


def test_submodule_clear_keeps_other_optimizer_group_gradients(device):
    first, second = _model(device), _model(device)
    optimizer = torch.optim.AdamW([first.weight, second.weight], lr=.01)
    first.weight.grad = torch.ones_like(first.weight)
    second.weight.grad = torch.full_like(second.weight, 2.)
    optimizer.step()
    before_first, before_second = _array(first.weight), _array(second.weight)
    retained = second.weight.grad
    first.zero_grad()
    assert first.weight.grad is None and second.weight.grad is retained
    optimizer.step()
    np.testing.assert_array_equal(_array(first.weight), before_first)
    assert not np.array_equal(_array(second.weight), before_second)
    assert int(optimizer.state[first.weight]["step"].item()) == 1
    assert int(optimizer.state[second.weight]["step"].item()) == 2


def test_clear_reaches_both_optimizers_sharing_a_parameter(device):
    model = _model(device)
    first = torch.optim.SGD(model.parameters(), lr=.1)
    second = torch.optim.AdamW(model.parameters(), lr=.01)
    model(torch.tensor([[2., 3.]], device=device)).sum().backward()
    before = _array(model.weight)
    model.zero_grad()
    first.step()
    second.step()
    assert model.weight.grad is None
    np.testing.assert_array_equal(_array(model.weight), before)
    assert model.weight not in second.state


@pytest.mark.parametrize("set_to_none", [True, False])
def test_multiple_updates_after_module_clear(device, set_to_none):
    model = _model(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=.125, momentum=.5)
    expected = np.array([[1., 2.]], dtype="float32")
    velocity = np.zeros_like(expected)
    for scale in (1., 2., .5, 3.):
        model.zero_grad(set_to_none=set_to_none)
        x = torch.tensor([[2. * scale, 3. * scale]], device=device)
        model(x).sum().backward()
        gradient = np.array([[2. * scale, 3. * scale]], dtype="float32")
        np.testing.assert_array_equal(_array(model.weight.grad), gradient)
        velocity = .5 * velocity + gradient
        expected -= .125 * velocity
        optimizer.step()
        np.testing.assert_array_equal(_array(model.weight), expected)
