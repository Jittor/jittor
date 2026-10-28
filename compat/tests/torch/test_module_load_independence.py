"""Default checkpoint copying keeps two models' allocations independent."""

import ctypes
from contextlib import contextmanager

import numpy as np
import pytest
import torch

from _optimizer_contract_support import require_optimizer_backend as require_optimizer_backend


DEVICES = ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='requires CUDA',
))]


def allocation(tensor, device):
    assert tensor.device.type == device
    if device == 'cuda':
        torch.cuda.synchronize()
        ordinal = ctypes.c_int(-1)
        driver = ctypes.CDLL('libcuda.so.1')
        assert driver.cuPointerGetAttribute(
            ctypes.byref(ordinal), 9, ctypes.c_void_p(tensor.data_ptr()),
        ) == 0
        assert ordinal.value == tensor.device.index
    return tensor.data_ptr(), tuple(tensor.stride()), tensor.dtype


def array(tensor):
    return tensor.detach().float().cpu().numpy().copy()


@contextmanager
def counted_native_fused_updates(device):
    calls = []
    if device != 'cuda' or not hasattr(torch, '_torch_compat_install_context'):
        yield calls
        return
    from jittor.backends.cuda.kernels.optim import fused_adamw_cuda
    from jittor._runtime.dispatch import override_kernel

    original = fused_adamw_cuda._cuda_fused_adamw_updates

    def counted(*args):
        calls.append(len(args[0]))
        return original(*args)

    with override_kernel('optim.adamw_fused', 'cuda', counted,
                         supports=fused_adamw_cuda._supports):
        yield calls


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('strided_target', [False, True])
def test_load_keeps_models_independent_under_adamw(device, dtype, strided_target):
    source = torch.nn.Linear(2, 2, device=device, dtype=dtype)
    target = torch.nn.Linear(2, 2, device=device, dtype=dtype)
    if strided_target:
        target.weight = torch.nn.Parameter(torch.zeros((2, 2), device=device, dtype=dtype).t())
    with torch.no_grad():
        source.weight.copy_(torch.tensor([[1., 2.], [3., 4.]], device=device, dtype=dtype))
        source.bias.copy_(torch.tensor([.25, .5], device=device, dtype=dtype))
    parameters = dict(target.named_parameters())
    before = {name: allocation(value, device) for name, value in parameters.items()}
    aliases = {name: value.detach() for name, value in parameters.items()}
    source_before = {name: allocation(value, device)
                     for name, value in source.named_parameters()}
    result = target.load_state_dict(source.state_dict())
    assert not result.missing_keys and not result.unexpected_keys
    snapshot = {name: array(value) for name, value in source.named_parameters()}
    for name, value in target.named_parameters():
        assert value is parameters[name] and value.is_leaf and value.requires_grad
        assert allocation(value, device) == before[name]
        assert allocation(aliases[name], device) == before[name]
        assert value.data_ptr() != source_before[name][0]
        np.testing.assert_array_equal(array(value), snapshot[name])
        np.testing.assert_array_equal(array(aliases[name]), snapshot[name])
    optimizer = torch.optim.AdamW(source.parameters(), lr=.125, weight_decay=.1,
                                 fused=device == 'cuda')
    with counted_native_fused_updates(device) as calls:
        source(torch.tensor([[1., 2.], [3., 4.]], device=device, dtype=dtype)).sum().backward()
        optimizer.step()
        if device == 'cuda':
            torch.cuda.synchronize()
    if device == 'cuda' and hasattr(torch, '_torch_compat_install_context'):
        assert calls == [2], 'The original regression requires an actual fused CUDA write'
    assert any(not np.array_equal(array(value), snapshot[name])
               for name, value in source.named_parameters())
    for name, value in target.named_parameters():
        assert allocation(value, device) == before[name]
        np.testing.assert_array_equal(array(value), snapshot[name])
        np.testing.assert_array_equal(array(aliases[name]), snapshot[name])
