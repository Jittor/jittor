"""Tensor gradients survive creation of an optimizer that consumes them."""
import numpy as np
import pytest
import torch

from _optimizer_contract_support import require_optimizer_backend as require_optimizer_backend


DEVICE_CASES = [
    'cpu',
    pytest.param('cuda', marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason='requires CUDA',
    )),
]


def numbers(tensor, device):
    assert tensor.device.type == device
    if hasattr(torch, '_torch_compat_install_context'):
        tensor.sync()
        assert tensor.location() == ('cpu' if device == 'cpu' else 'device')
    return tensor.detach().float().cpu().numpy().copy()


def make_optimizer(name, parameters):
    options = {'lr': .125}
    if name != 'RMSprop':
        options['weight_decay'] = 0
    return getattr(torch.optim, name)(parameters, **options)


@pytest.mark.parametrize('name', ['SGD', 'Adam', 'AdamW', 'RMSprop'])
@pytest.mark.parametrize('source', ['manual', 'backward'])
@pytest.mark.parametrize('device', DEVICE_CASES)
def test_constructor_keeps_gradients_and_update_state(name, source, device):
    trajectories = []
    for before_constructor in (False, True):
        parameter = torch.nn.Parameter(torch.tensor([1., -2.], device=device))
        gradient = torch.tensor([.5, -.25], device=device)

        def prepare():
            if source == 'manual':
                parameter.grad = gradient.clone()
            else:
                (parameter * gradient).sum().backward()

        if before_constructor:
            prepare()
        published = parameter.grad
        optimizer = make_optimizer(name, [parameter])
        if before_constructor:
            assert parameter.grad is published
        else:
            prepare()
            published = parameter.grad
        snapshots = []
        for _ in range(2):
            optimizer.step()
            assert parameter.grad is published
            np.testing.assert_array_equal(numbers(parameter.grad, device), [.5, -.25])
            snapshots.append(numbers(parameter, device))
        assert not np.array_equal(snapshots[0], [1., -2.])
        if name in ('Adam', 'AdamW'):
            assert float(optimizer.state[parameter]['step']) == 2
        trajectories.append(snapshots)
    np.testing.assert_array_equal(trajectories[0], trajectories[1])


@pytest.mark.parametrize('device', DEVICE_CASES)
def test_constructor_keeps_shared_gradient_storage_and_clear(device):
    parameter = torch.nn.Parameter(torch.tensor([1., -2.], device=device))
    published = torch.tensor([1., -.5], device=device)
    parameter.grad = published
    first = torch.optim.SGD([parameter], lr=.125)
    second = torch.optim.SGD([parameter], lr=.125)
    with torch.no_grad():
        published.mul_(.5)
    assert parameter.grad is published
    first.step()
    second.step()
    np.testing.assert_array_equal(numbers(parameter, device), [.875, -1.9375])
    first.zero_grad(set_to_none=True)
    assert parameter.grad is None
    before = numbers(parameter, device)
    first.step()
    second.step()
    np.testing.assert_array_equal(numbers(parameter, device), before)


@pytest.mark.parametrize('name', ['SGD', 'Adam', 'AdamW', 'RMSprop'])
@pytest.mark.parametrize('device', DEVICE_CASES)
def test_constructor_preserves_missing_gradients_in_other_groups(name, device):
    used = torch.nn.Parameter(torch.tensor([1., -2.], device=device))
    unused = torch.nn.Parameter(torch.tensor([3., 4.], device=device))
    used.grad = torch.tensor([.5, -.25], device=device)
    optimizer = make_optimizer(name, [{'params': [used]}, {'params': [unused]}])
    optimizer.step()
    assert unused.grad is None
    np.testing.assert_array_equal(numbers(unused, device), [3., 4.])
    assert not np.array_equal(numbers(used, device), [1., -2.])


@pytest.mark.parametrize('name', ['SGD', 'AdamW'])
@pytest.mark.parametrize('device', DEVICE_CASES)
def test_existing_zero_gradient_applies_weight_decay(name, device):
    parameter = torch.nn.Parameter(torch.tensor([1., -2.], device=device))
    parameter.grad = torch.zeros_like(parameter)
    optimizer = getattr(torch.optim, name)([parameter], lr=.125, weight_decay=.5)
    optimizer.step()
    np.testing.assert_array_equal(numbers(parameter, device), [.9375, -1.875])


@pytest.mark.parametrize('name', ['SGD', 'Adam', 'AdamW', 'RMSprop'])
@pytest.mark.parametrize('device', DEVICE_CASES)
def test_no_existing_gradient_leaves_parameter_unchanged(name, device):
    parameter = torch.nn.Parameter(torch.tensor([1., -2.], device=device))
    optimizer = make_optimizer(name, [parameter])
    optimizer.step()
    assert parameter.grad is None
    np.testing.assert_array_equal(numbers(parameter, device), [1., -2.])


@pytest.mark.parametrize('name', ['Adam', 'AdamW'])
@pytest.mark.parametrize('foreach', [None, False])
@pytest.mark.parametrize('device', DEVICE_CASES)
def test_adam_update_preserves_parameter_leaf_and_reversible_freeze(name, foreach, device):
    parameter = torch.nn.Parameter(torch.tensor([1., -2.], device=device))
    optimizer = getattr(torch.optim, name)(
        [parameter], lr=.125, weight_decay=.02, foreach=foreach,
    )
    gradient = torch.tensor([.5, -.25], device=device)
    parameter.grad = gradient
    before = numbers(parameter, device)
    for _ in range(2):
        optimizer.step()
        assert optimizer.param_groups[0]['params'][0] is parameter
        assert isinstance(parameter, torch.nn.Parameter)
        assert parameter.grad is gradient
        assert parameter.requires_grad and parameter.is_leaf
        assert parameter.grad_fn is None
        parameter.requires_grad_(False)
        assert not parameter.requires_grad and parameter.is_leaf
        parameter.requires_grad_(True)
        assert parameter.requires_grad and parameter.is_leaf
        assert parameter.grad_fn is None
        for key in ('exp_avg', 'exp_avg_sq'):
            moment = optimizer.state[parameter][key]
            assert not moment.requires_grad and moment.is_leaf
            assert moment.grad_fn is None
        intermediate = parameter * 2
        assert not intermediate.is_leaf and intermediate.grad_fn is not None
        with pytest.raises(RuntimeError, match='only change requires_grad flags of leaf'):
            intermediate.requires_grad_(False)
        assert intermediate.requires_grad and not intermediate.is_leaf
    assert not np.array_equal(numbers(parameter, device), before)


@pytest.mark.parametrize('name', ['Adam', 'AdamW'])
@pytest.mark.parametrize('outer_grad_enabled', [True, False])
@pytest.mark.parametrize('device', DEVICE_CASES)
def test_adam_closure_enables_grad_and_restores_outer_mode(name, outer_grad_enabled, device):
    parameter = torch.nn.Parameter(torch.tensor([1., -2.], device=device))
    optimizer = getattr(torch.optim, name)([parameter], lr=.125, foreach=False)
    losses = []

    def closure():
        assert torch.is_grad_enabled()
        optimizer.zero_grad()
        loss = parameter.square().sum()
        loss.backward()
        losses.append(loss)
        return loss

    with torch.set_grad_enabled(outer_grad_enabled):
        result = optimizer.step(closure)
        assert torch.is_grad_enabled() is outer_grad_enabled
    assert result is losses[0] and len(losses) == 1
    assert parameter.requires_grad and parameter.is_leaf
    assert parameter.grad_fn is None and parameter.grad is not None
    assert not np.array_equal(numbers(parameter, device), [1., -2.])


@pytest.mark.parametrize('name', ['Adam', 'AdamW'])
@pytest.mark.parametrize('outer_grad_enabled', [True, False])
@pytest.mark.parametrize('device', DEVICE_CASES)
def test_adam_closure_exception_restores_grad_mode_without_update(name, outer_grad_enabled, device):
    parameter = torch.nn.Parameter(torch.tensor([1., -2.], device=device))
    optimizer = getattr(torch.optim, name)([parameter], lr=.125, foreach=False)
    calls = []
    error = ValueError('closure failed before optimizer update')

    def closure():
        calls.append(torch.is_grad_enabled())
        raise error

    with torch.set_grad_enabled(outer_grad_enabled):
        with pytest.raises(ValueError, match='closure failed before optimizer update') as caught:
            optimizer.step(closure)
        assert torch.is_grad_enabled() is outer_grad_enabled
    assert caught.value is error and calls == [True]
    np.testing.assert_array_equal(numbers(parameter, device), [1., -2.])
    assert parameter.requires_grad and parameter.is_leaf
    assert parameter.grad_fn is None and parameter.grad is None
