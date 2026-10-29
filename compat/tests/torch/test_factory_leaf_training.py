"""Torch constant factories create trainable independent leaves."""
import numpy as np
import pytest
import torch


def make_constant(name, template, requires_grad):
    kwargs = {"requires_grad": requires_grad}
    if name.endswith("_like"):
        args = (template, 2.0) if name == "full_like" else (template,)
    else:
        args = ((3,), 2.0) if name == "full" else ((3,),)
    return getattr(torch, name)(*args, **kwargs)


@pytest.mark.parametrize("name", ["ones", "zeros", "full", "ones_like", "zeros_like", "full_like"])
@pytest.mark.parametrize("construction_grad", [False, True])
def test_constant_factory_leaf_and_gradient(name, construction_grad):
    template = torch.ones(3, requires_grad=True)
    x = make_constant(name, template, construction_grad)
    if not construction_grad:
        x.requires_grad_(True)
    assert x.is_leaf and x.grad_fn is None and x.requires_grad
    (x * 3).sum().backward()
    np.testing.assert_array_equal(x.grad.numpy(), np.full((3,), 3.0))
    assert template.grad is None


@pytest.mark.parametrize("name", ["ones_like", "zeros_like", "full_like"])
def test_like_factory_is_independent_from_template(name):
    template = torch.ones(3, dtype=torch.float64, requires_grad=True)
    x = make_constant(name, template, True)
    assert x.is_leaf and x.dtype == template.dtype and x.device == template.device
    assert x.data_ptr() != template.data_ptr()
    assert torch.autograd.grad(x.sum(), template, allow_unused=True)[0] is None


def test_factory_leaf_sgd_updates_and_no_grad_creation():
    with torch.no_grad():
        x = torch.ones(3, requires_grad=True)
    assert x.is_leaf and x.requires_grad
    optimizer = torch.optim.SGD([x], lr=0.1)
    (x * 3).sum().backward()
    optimizer.step()
    np.testing.assert_allclose(x.detach().numpy(), np.full((3,), 0.7), rtol=0, atol=1e-6)


def test_factory_leaf_supports_second_order_gradient():
    x = torch.full((3,), 2.0, requires_grad=True)
    first = torch.autograd.grad((x ** 3).sum(), x, create_graph=True)[0]
    second = torch.autograd.grad(first.sum(), x)[0]
    np.testing.assert_allclose(first.detach().numpy(), np.full((3,), 12.0), rtol=0, atol=1e-6)
    np.testing.assert_allclose(second.detach().numpy(), np.full((3,), 12.0), rtol=0, atol=1e-6)
