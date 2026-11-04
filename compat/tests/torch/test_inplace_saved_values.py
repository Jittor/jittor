"""Legal in-place backward retains the old values its derivative needs."""
import os

import numpy as np
import pytest
import torch


@pytest.fixture
def device():
    return os.environ.get("JITTOR_TEST_DEVICES", "cpu").split(",")[0]


@pytest.mark.parametrize("name", [
    "erf", "log", "exp", "sqrt", "abs", "sigmoid", "tanh",
    "reciprocal", "rsqrt", "clamp",
])
def test_inplace_saved_old_input(device, name):
    values = [-.5, 0., 3.] if name == "abs" else [.5, 2., 3.]
    source = torch.tensor(values, device=device, requires_grad=True)
    reference = source.detach().clone().requires_grad_(True)
    args = (1., 2.5) if name == "clamp" else ()
    actual = source * 1
    assert getattr(actual, name + "_")(*args) is actual
    expected = getattr(reference, name)(*args)
    actual_gradient, = torch.autograd.grad(actual.sum(), source)
    expected_gradient, = torch.autograd.grad(expected.sum(), reference)
    assert actual.device.type == actual_gradient.device.type == device
    np.testing.assert_allclose(actual.detach().cpu().numpy(),
                               expected.detach().cpu().numpy(), rtol=1e-5)
    np.testing.assert_allclose(actual_gradient.detach().cpu().numpy(),
                               expected_gradient.detach().cpu().numpy(), rtol=1e-5)


def test_multiply_alias_saves_the_old_destination(device):
    source = torch.tensor([.2, .3], device=device, requires_grad=True)
    result = source * 1
    assert result.multiply_(source) is result
    gradient, = torch.autograd.grad(result.sum(), source)
    np.testing.assert_allclose(gradient.detach().cpu().numpy(), [.4, .6], rtol=1e-6)
