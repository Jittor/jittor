"""Optimizer writes retain real parameter/state allocations and live aliases."""

import ctypes

import numpy as np
import pytest
import torch

from _optimizer_contract_support import require_optimizer_backend as require_optimizer_backend


DEVICES = ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='requires CUDA',
))]
ALGORITHMS = ['sgd', 'sgd_momentum', 'sgd_nesterov', 'adam', 'adamw', 'rmsprop',
              'sgd_fused', 'sgd_momentum_fused', 'sgd_nesterov_fused', 'adamw_fused']


def make_optimizer(parameter, algorithm):
    fused = algorithm.endswith('_fused')
    kind = algorithm[:-6] if fused else algorithm
    if kind.startswith('sgd'):
        options = ({'momentum': .5, 'weight_decay': .25, 'dampening': .25}
                   if kind == 'sgd_momentum' else
                   {'momentum': .5, 'weight_decay': .25, 'nesterov': True}
                   if kind == 'sgd_nesterov' else {})
        return torch.optim.SGD([parameter], lr=.125, foreach=False, fused=fused, **options)
    if kind == 'rmsprop':
        return torch.optim.RMSprop([parameter], lr=.125, eps=.125, alpha=.5,
                                   weight_decay=.25)
    cls = torch.optim.AdamW if kind == 'adamw' else torch.optim.Adam
    options = {'fused': fused} if kind == 'adamw' else {}
    return cls([parameter], lr=.125, eps=.125, betas=(.5, .5),
               weight_decay=.25, foreach=False, **options)


def allocation(tensor, device):
    assert tensor.device.type == device
    if device == 'cuda':
        torch.cuda.synchronize()
        ordinal = ctypes.c_int(-1)
        assert ctypes.CDLL('libcuda.so.1').cuPointerGetAttribute(
            ctypes.byref(ordinal), 9, ctypes.c_void_p(tensor.data_ptr()),
        ) == 0
        assert ordinal.value == tensor.device.index
    return tensor.data_ptr(), tuple(tensor.stride()), tensor.dtype


def array(tensor):
    detached = tensor.detach()
    detached = detached.double() if tensor.dtype == torch.float64 else detached.float()
    return detached.cpu().numpy().copy()


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64, torch.bfloat16])
@pytest.mark.parametrize('strided', [False, True])
@pytest.mark.parametrize('algorithm', ALGORITHMS)
def test_optimizer_preserves_parameter_and_state_storage(device, dtype, strided, algorithm):
    raw = torch.tensor([[1., 2.], [3., 4.]], dtype=dtype, device=device)
    parameter = torch.nn.Parameter(raw.t() if strided else raw)
    aliases = [parameter.detach(), parameter.data, parameter.detach().t()]
    before = allocation(parameter, device)
    geometry = [allocation(value, device) for value in aliases]
    optimizer = make_optimizer(parameter, algorithm)
    state_aliases = {}
    for scale in (1., .5, 2.):
        optimizer.zero_grad()
        old = array(parameter)
        gradient = torch.tensor([[.25, .5], [.75, 1.]], dtype=dtype, device=device) * scale
        (parameter * gradient).sum().backward()
        published = parameter.grad
        grad_values = array(published)
        optimizer.step()
        # Read the aliases first: observing the parameter must not be what
        # makes a pending optimizer write visible to an existing alias.
        observed = [array(value) for value in aliases]
        current = array(parameter)
        assert np.any(current != old)
        assert allocation(parameter, device) == before
        assert parameter.is_leaf and parameter.requires_grad
        for index, (value, actual) in enumerate(zip(aliases, observed)):
            assert allocation(value, device) == geometry[index]
            np.testing.assert_array_equal(actual, current.T if index == 2 else current)
        assert parameter.grad is published
        np.testing.assert_array_equal(array(published), grad_values)
        for name, value in optimizer.state.get(parameter, {}).items():
            # Tensor-valued moments are real mutable buffers. Step counters
            # have a separate state ABI and are not parameter-shaped storage.
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(parameter.shape):
                continue
            if name in state_aliases:
                identity, alias, saved_geometry = state_aliases[name]
                assert value is identity
                assert allocation(value, device) == saved_geometry
                assert allocation(alias, device) == saved_geometry
                np.testing.assert_array_equal(array(alias), array(value))
            else:
                state_aliases[name] = value, value.detach(), allocation(value, device)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('algorithm', ['sgd', 'sgd_fused', 'adamw', 'adamw_fused'])
def test_optimizer_retains_old_independent_forward_output(device, algorithm):
    parameter = torch.nn.Parameter(torch.tensor([[0., 0.], [0., 0.]], device=device))
    optimizer = make_optimizer(parameter, algorithm)
    old_output = parameter.exp().sum()
    # Leave old_output unobserved before the store. Its exp backward saves
    # that independent output, whereas input-saving mutation guards are a
    # separate native autograd contract.
    (parameter * torch.tensor([[.25, .5], [.75, 1.]], device=device)).sum().backward()
    optimizer.step()
    optimizer.zero_grad()
    old_output.backward()
    allocation(parameter.grad, device)
    np.testing.assert_array_equal(array(parameter.grad), np.ones((2, 2)))


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('algorithm', ['sgd', 'sgd_fused', 'adamw', 'adamw_fused'])
def test_optimizer_retains_old_transitive_forward_output(device, algorithm):
    parameter = torch.nn.Parameter(torch.tensor([[0., 0.], [0., 0.]], device=device))
    optimizer = make_optimizer(parameter, algorithm)
    # The multiply consumes exp's output, so the saved value is one hop away
    # from the parameter storage the optimizer rewrites.
    old_output = (parameter.exp() * 2).sum()
    (parameter * torch.tensor([[.25, .5], [.75, 1.]], device=device)).sum().backward()
    optimizer.step()
    optimizer.zero_grad()
    old_output.backward()
    allocation(parameter.grad, device)
    np.testing.assert_array_equal(array(parameter.grad), np.full((2, 2), 2.0))
