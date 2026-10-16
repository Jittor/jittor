"""Checkpoint value copies preserve the caller's grad mode and actual storage.

Also runs unchanged in an independent genuine Torch interpreter. General saved
input-version rejection after mutation is a separate native autograd contract.
"""

import ctypes
import os

import numpy as np
import pytest
import torch


@pytest.fixture(params=os.environ.get('JITTOR_TEST_DEVICES', 'cpu').split(','))
def device(request):
    assert request.param in ('cpu', 'cuda')
    if request.param == 'cuda':
        assert torch.cuda.is_available(), 'Requested real CUDA is required'
    return request.param


def allocated_on(tensor, device):
    assert tensor.device.type == device
    if device == 'cuda' and tensor.numel():
        torch.cuda.synchronize()
        ordinal = ctypes.c_int(-1)
        driver = ctypes.CDLL('libcuda.so.1')
        assert driver.cuPointerGetAttribute(ctypes.byref(ordinal), 9,
                                            ctypes.c_void_p(tensor.data_ptr())) == 0
        assert ordinal.value == tensor.device.index


@pytest.mark.parametrize('enabled', (True, False))
@pytest.mark.parametrize('dtype', (torch.float32, torch.float64))
def test_load_copies_without_recording_edges_or_changing_caller_mode(device, enabled, dtype):
    with torch.set_grad_enabled(enabled):
        model = torch.nn.Linear(3, 2).to(device=device, dtype=dtype)
        model.bias.requires_grad_(False)
        model.register_buffer('count', torch.tensor([1, 2], device=device))
        weight, bias, count = model.weight, model.bias, model.count
        old_weight = weight.detach().clone()
        old_values = weight.detach().cpu().numpy().copy()
        alias = weight.detach()
        buffer_alias = count.detach()
        pointers = (weight.data_ptr(), bias.data_ptr(), count.data_ptr())
        incoming = torch.tensor(np.full((2, 3), .25), dtype=torch.float64, requires_grad=True)
        incoming_bias = torch.tensor([.5, .5], dtype=torch.float64, requires_grad=True)
        result = model.load_state_dict({'weight': incoming, 'bias': incoming_bias,
                                       'count': torch.tensor([7, 8], dtype=torch.int32)})
        assert torch.is_grad_enabled() is enabled
        assert not result.missing_keys and not result.unexpected_keys
        assert model.weight is weight and model.bias is bias and model.count is count
        assert pointers == (weight.data_ptr(), bias.data_ptr(), count.data_ptr())
        assert weight.requires_grad and weight.is_leaf and weight.grad_fn is None
        assert not bias.requires_grad and weight.dtype == bias.dtype == dtype
        assert count.dtype == torch.int64
        for value in (weight, bias, count, alias, buffer_alias):
            allocated_on(value, device)
        np.testing.assert_array_equal(old_weight.cpu().numpy(), old_values)
        np.testing.assert_array_equal(alias.cpu().numpy(), np.full((2, 3), .25))
        np.testing.assert_array_equal(buffer_alias.cpu().numpy(), [7, 8])
    with torch.enable_grad():
        weight.square().sum().backward()
    np.testing.assert_array_equal(weight.grad.cpu().numpy(), np.full((2, 3), .5))
    assert bias.grad is None and incoming.grad is None and incoming_bias.grad is None


@pytest.mark.parametrize('enabled', (True, False))
def test_invalid_load_preserves_grad_mode_and_original_values(device, enabled):
    with torch.set_grad_enabled(enabled):
        model = torch.nn.Linear(3, 2).to(device)
        weight = model.weight
        values = weight.detach().cpu().numpy().copy()
        pointer = weight.data_ptr()
        with pytest.raises(RuntimeError, match='size mismatch'):
            model.load_state_dict({'weight': torch.tensor([1.], device=device),
                                   'bias': torch.tensor([1., 1.], device=device)})
        assert torch.is_grad_enabled() is enabled
        assert model.weight is weight and weight.data_ptr() == pointer
        np.testing.assert_array_equal(weight.detach().cpu().numpy(), values)


def test_old_constant_gradient_consumer_remains_valid_after_load(device):
    # Torch permits this consumer: its backward does not save the old input.
    model = torch.nn.Linear(3, 2).to(device)
    weight = model.weight
    old = (weight * 3).sum()
    model.load_state_dict({key: torch.full_like(value, .25)
                           for key, value in model.state_dict().items()})
    old.backward()
    allocated_on(weight.grad, device)
    np.testing.assert_array_equal(weight.grad.cpu().numpy(), np.full((2, 3), 3.))
