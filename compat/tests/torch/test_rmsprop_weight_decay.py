"""RMSprop couples decay into both the update and square-average state."""

import copy

import numpy as np
import pytest
import torch

from _optimizer_contract_support import require_optimizer_backend as require_optimizer_backend


DEVICES = ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='requires CUDA',
))]


def optimizer_array(value, device):
    assert value.device.type == device
    if hasattr(torch, '_torch_compat_install_context'):
        value.sync()
        assert value.location() == ('cpu' if device == 'cpu' else 'device')
    return value.detach().cpu().numpy().copy()


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('global_decay,group_decay', [
    (0., None), (.25, None), (.25, 0.), (0., .5), (None, .5),
])
def test_rmsprop_decay_updates_parameter_and_square_average(
        device, dtype, global_decay, group_decay):
    parameter = torch.nn.Parameter(torch.tensor([1., -2., .5], dtype=dtype, device=device))
    group = {'params': [parameter]}
    if group_decay is not None:
        group['weight_decay'] = group_decay
    options = {} if global_decay is None else {'weight_decay': global_decay}
    optimizer = torch.optim.RMSprop(
        [group], lr=.03125, alpha=.875, eps=.125, **options,
    )
    expected = optimizer_array(parameter, device)
    square = np.zeros_like(expected)
    decay = global_decay if group_decay is None else group_decay
    tolerance = 2e-6 if dtype == torch.float32 else 1e-12
    for step, gradient in enumerate(([.5, -.25, .125], [-.25, .5, 0.], [.125, 0., -.5]), 1):
        parameter.grad = torch.tensor(gradient, dtype=dtype, device=device)
        published = parameter.grad
        effective = np.asarray(gradient, dtype=expected.dtype) + decay * expected
        square = .875 * square + .125 * effective * effective
        expected -= .03125 * effective / (np.sqrt(square) + .125)
        optimizer.step()
        assert parameter.grad is published
        np.testing.assert_array_equal(optimizer_array(parameter.grad, device), gradient)
        np.testing.assert_allclose(optimizer_array(parameter, device), expected,
                                   rtol=tolerance, atol=tolerance)
        state = optimizer.state[parameter]
        np.testing.assert_allclose(optimizer_array(state['square_avg'], device), square,
                                   rtol=tolerance, atol=tolerance)
        assert float(state['step']) == step
        assert optimizer.state_dict()['param_groups'][0]['weight_decay'] == decay


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_rmsprop_decay_checkpoint_restores_group_and_continuation(device, dtype):
    parameter = torch.nn.Parameter(torch.tensor([1., -2.], dtype=dtype, device=device))
    optimizer = torch.optim.RMSprop([parameter], lr=.03125, weight_decay=.25)
    parameter.grad = torch.tensor([.5, -.25], dtype=dtype, device=device)
    optimizer.step()
    saved = copy.deepcopy(optimizer.state_dict())
    assert saved['param_groups'][0]['weight_decay'] == .25
    restored = torch.nn.Parameter(parameter.detach().clone())
    resumed = torch.optim.RMSprop([restored], lr=.5, weight_decay=0.)
    resumed.load_state_dict(saved)
    for gradient in ([-.25, .5], [.125, -.125]):
        parameter.grad = torch.tensor(gradient, dtype=dtype, device=device)
        restored.grad = parameter.grad.clone()
        optimizer.step()
        resumed.step()
        np.testing.assert_array_equal(optimizer_array(restored, device),
                                      optimizer_array(parameter, device))
        np.testing.assert_array_equal(optimizer_array(resumed.state[restored]['square_avg'], device),
                                      optimizer_array(optimizer.state[parameter]['square_avg'], device))
        assert float(resumed.state[restored]['step']) == float(optimizer.state[parameter]['step'])


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('missing', [False, True])
def test_rmsprop_zero_gradient_decays_but_missing_gradient_skips(device, missing):
    parameter = torch.nn.Parameter(torch.tensor([1., -2.], device=device))
    optimizer = torch.optim.RMSprop([parameter], lr=.03125, weight_decay=.25)
    if not missing:
        parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
    actual = optimizer_array(parameter, device)
    assert bool(np.array_equal(actual, [1., -2.])) == missing
    assert (parameter in optimizer.state) != missing


@pytest.mark.parametrize('invalid', [-.25, float('nan')])
def test_rmsprop_rejects_invalid_global_decay(invalid):
    parameter = torch.nn.Parameter(torch.tensor([1.]))
    with pytest.raises(ValueError, match='weight_decay'):
        torch.optim.RMSprop([parameter], weight_decay=invalid)
