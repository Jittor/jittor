"""SGD execution options select existing native owners and survive checkpoints."""

import ctypes
import io
from contextlib import contextmanager
from itertools import product

import numpy as np
import pytest
import torch

from _optimizer_contract_support import require_optimizer_backend as require_optimizer_backend


DEVICES = ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='requires CUDA',
))]
OPTIONS = [(foreach, fused) for foreach, fused in product((None, False, True), repeat=2)
           if not (foreach is True and fused is True)]
CONFIGURATIONS = {
    'plain': {},
    'momentum': {'momentum': .5, 'dampening': .25, 'weight_decay': .25},
    'nesterov': {'momentum': .5, 'nesterov': True, 'weight_decay': .25},
}
# Independent Torch 2.13 CPU execution; CUDA checks use the same dyadic inputs.
_GOLDEN = {'plain/0/0/p': [0.9375, -1.96875],
 'plain/0/1/p': [[0.1875, 0.53125], [0.984375, 1.96875]],
 'plain/1/0/p': [0.90625, -1.953125],
 'plain/1/1/p': [[0.15625, 0.546875], [0.9765625, 1.953125]],
 'plain/2/0/p': [0.78125, -1.890625],
 'plain/2/1/p': [[0.03125, 0.609375], [0.9453125, 1.890625]],
 'momentum/0/0/p': [0.90625, -1.90625],
 'momentum/0/0/m': [0.75, -0.75],
 'momentum/0/1/p': [[0.1796875, 0.515625], [0.953125, 1.90625]],
 'momentum/0/1/m': [[0.5625, -0.125], [0.375, 0.75]],
 'momentum/1/0/p': [0.814697265625, -1.802978515625],
 'momentum/1/0/m': [0.732421875, -0.826171875],
 'momentum/1/1/p': [[0.11688232421875, 0.5230712890625],
                    [0.9014892578125, 1.802978515625]],
 'momentum/1/1/m': [[0.50244140625, -0.0595703125],
                    [0.4130859375, 0.826171875]],
 'momentum/2/0/p': [0.6560764312744141, -1.662210464477539],
 'momentum/2/0/m': [1.2689666748046875, -1.1261444091796875],
 'momentum/2/1/p': [[-0.011009693145751953, 0.5614099502563477],
                    [0.8311052322387695, 1.662210464477539]],
 'momentum/2/1/m': [[1.0231361389160156, -0.30670928955078125],
                    [0.5630722045898438, 1.1261444091796875]],
 'nesterov/0/0/p': [0.859375, -1.859375],
 'nesterov/0/0/m': [0.75, -0.75],
 'nesterov/0/1/p': [[0.14453125, 0.5234375], [0.9296875, 1.859375]],
 'nesterov/0/1/m': [[0.5625, -0.125], [0.375, 0.75]],
 'nesterov/1/0/p': [0.748779296875, -1.725341796875],
 'nesterov/1/0/m': [0.83984375, -0.96484375],
 'nesterov/1/1/p': [[0.07330322265625, 0.5262451171875],
                    [0.8626708984375, 1.725341796875]],
 'nesterov/1/1/m': [[0.5673828125, -0.056640625], [0.482421875, 0.96484375]],
 'nesterov/2/0/p': [0.4999351501464844, -1.5205650329589844],
 'nesterov/2/0/m': [1.60711669921875, -1.41375732421875],
 'nesterov/2/1/p': [[-0.13536357879638672, 0.5970973968505859],
                    [0.7602825164794922, 1.5205650329589844]],
 'nesterov/2/1/m': [[1.3020172119140625, -0.396759033203125],
                    [0.706878662109375, 1.41375732421875]]}


def allocated(tensor, device):
    assert tensor.device.type == device
    if device == 'cuda':
        torch.cuda.synchronize()
        ordinal = ctypes.c_int(-1)
        assert ctypes.CDLL('libcuda.so.1').cuPointerGetAttribute(
            ctypes.byref(ordinal), 9, ctypes.c_void_p(tensor.data_ptr()),
        ) == 0
        assert ordinal.value == tensor.device.index


def values(tensor):
    return tensor.detach().float().cpu().numpy().copy()


def parameters(device):
    return [torch.tensor([1., -2.], device=device, requires_grad=True),
            torch.tensor([[.25, .5], [1., 2.]], device=device, requires_grad=True)]


def advance(params, optimizer, scale, device):
    optimizer.zero_grad()
    gradients = [torch.tensor([.5, -.25], device=device) * scale,
                 torch.tensor([[.5, -.25], [.125, .25]], device=device) * scale]
    sum((p * g).sum() for p, g in zip(params, gradients)).backward()
    published = [p.grad for p in params]
    snapshots = [values(g) for g in published]
    optimizer.step()
    for parameter, gradient, expected in zip(params, published, snapshots):
        assert parameter.grad is gradient
        np.testing.assert_array_equal(values(gradient), expected)
        allocated(parameter, device)
        allocated(gradient, device)
        momentum = optimizer.state.get(parameter, {}).get('momentum_buffer')
        if momentum is not None:
            allocated(momentum, device)


@contextmanager
def native_launches(device):
    calls = []
    if device != 'cuda' or not hasattr(torch, '_torch_compat_install_context'):
        yield calls
        return
    from jittor._runtime.dispatch import override_kernel
    from jittor.backends.cuda.kernels.optim import fused_sgd_cuda
    original = fused_sgd_cuda._fused_sgd_cuda

    def counted(*args, **kwargs):
        calls.append(len(args[0]))
        return original(*args, **kwargs)

    with override_kernel('optim.sgd_fused', 'cuda', counted,
                         supports=fused_sgd_cuda._supports_fused_sgd):
        yield calls


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('configuration', CONFIGURATIONS)
@pytest.mark.parametrize('foreach,fused', OPTIONS)
def test_sgd_execution_options_and_checkpoint(device, configuration, foreach, fused):
    params = parameters(device)
    optimizer = torch.optim.SGD(params, lr=.125, foreach=foreach, fused=fused,
                                **CONFIGURATIONS[configuration])
    with native_launches(device) as calls:
        for step, scale in enumerate((1., .5, 2.)):
            advance(params, optimizer, scale, device)
            for index, parameter in enumerate(params):
                np.testing.assert_array_equal(values(parameter),
                    _GOLDEN[f'{configuration}/{step}/{index}/p'])
                momentum = optimizer.state.get(parameter, {}).get('momentum_buffer')
                if configuration == 'plain':
                    assert momentum is None
                else:
                    np.testing.assert_array_equal(values(momentum),
                        _GOLDEN[f'{configuration}/{step}/{index}/m'])
        state = optimizer.state_dict()
        assert state['param_groups'][0]['foreach'] is foreach
        assert state['param_groups'][0]['fused'] is fused
        stream = io.BytesIO()
        torch.save(state, stream)
        restored = [torch.tensor(values(p), device=device, requires_grad=True) for p in params]
        resumed = torch.optim.SGD(restored, lr=.5)
        resumed.load_state_dict(torch.load(io.BytesIO(stream.getvalue()), weights_only=False))
        assert resumed.state_dict()['param_groups'][0]['foreach'] is foreach
        assert resumed.state_dict()['param_groups'][0]['fused'] is fused
        for ps, opt in ((params, optimizer), (restored, resumed)):
            advance(ps, opt, .25, device)
        for left, right in zip(params, restored):
            np.testing.assert_array_equal(values(left), values(right))
            if configuration != 'plain':
                np.testing.assert_array_equal(values(optimizer.state[left]['momentum_buffer']),
                                             values(resumed.state[right]['momentum_buffer']))
    if device == 'cuda' and hasattr(torch, '_torch_compat_install_context'):
        expected = [2] * 5 if fused is not False and (fused is True or foreach is not False) else []
        assert calls == expected


@pytest.mark.parametrize('device', DEVICES)
def test_sgd_rejects_conflicting_execution_options(device):
    with pytest.raises(RuntimeError, match='fused.*foreach|foreach.*fused'):
        torch.optim.SGD(parameters(device), lr=.125, foreach=True, fused=True)


@pytest.mark.parametrize('device', DEVICES)
def test_sgd_group_options_override_constructor(device):
    params = parameters(device)
    optimizer = torch.optim.SGD([
        {'params': [params[0]], 'foreach': False},
        {'params': [params[1]], 'foreach': None, 'fused': True},
    ], lr=.125, foreach=True, fused=False)
    with native_launches(device) as calls:
        advance(params, optimizer, 1., device)
    if device == 'cuda' and hasattr(torch, '_torch_compat_install_context'):
        assert calls == [1]
    groups = optimizer.state_dict()['param_groups']
    assert groups[0]['foreach'] is False and groups[0]['fused'] is False
    assert groups[1]['foreach'] is None and groups[1]['fused'] is True
    for index, parameter in enumerate(params):
        np.testing.assert_array_equal(values(parameter), _GOLDEN[f'plain/0/{index}/p'])


@pytest.mark.parametrize('device', DEVICES)
def test_sgd_added_group_exports_inherited_options(device):
    params = parameters(device)
    optimizer = torch.optim.SGD([params[0]], lr=.125, foreach=True, fused=False)
    optimizer.add_param_group({'params': [params[1]]})
    groups = optimizer.state_dict()['param_groups']
    assert len(groups) == 2
    assert all(group['foreach'] is True and group['fused'] is False for group in groups)
