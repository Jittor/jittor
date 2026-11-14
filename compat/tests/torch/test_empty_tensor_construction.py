"""Empty Python sequences retain tensor factory dtype, leaf and device rules."""
import numpy as np
import pytest
import torch

from _rng_init_support import preserve_random_streams, require_rng_device


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('sequence', [list, tuple])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64, torch.bfloat16,
                                  torch.int64, torch.complex128])
def test_empty_sequence_metadata_and_leaf(device, sequence, dtype):
    require_rng_device(device)
    with preserve_random_streams():
        floating = dtype in (torch.float32, torch.float64, torch.bfloat16, torch.complex128)
        with torch.no_grad():
            value = torch.tensor(sequence(), dtype=dtype, device=device,
                                 requires_grad=floating)
        assert tuple(value.shape) == (0,) and value.numel() == 0
        assert value.dtype == dtype and value.device.type == device
        assert value.is_leaf and value.grad_fn is None
        assert value.requires_grad == floating


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('default_dtype', [torch.float32, torch.float64])
def test_empty_sequence_uses_default_dtype_and_device(device, default_dtype):
    require_rng_device(device)
    previous = torch.get_default_dtype()
    with preserve_random_streams():
        try:
            torch.set_default_dtype(default_dtype)
            with torch.device(device):
                value = torch.tensor([])
            assert value.dtype == default_dtype and value.device.type == device
            assert not value.requires_grad and value.is_leaf
        finally:
            torch.set_default_dtype(previous)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_empty_sequence_concatenates_with_real_device_values(device):
    require_rng_device(device)
    with preserve_random_streams():
        empty = torch.tensor([], dtype=torch.float32, device=device)
        other = torch.tensor([], dtype=torch.float32, device=device)
        assert empty is not other
        values = torch.arange(6, dtype=torch.float32, device=device).reshape(2, 3)
        result = torch.cat([empty, values], dim=1)
        assert result.device.type == device
        np.testing.assert_array_equal(result.cpu().numpy(), np.arange(6).reshape(2, 3))


def test_empty_integer_cannot_require_gradient():
    with pytest.raises(RuntimeError):
        torch.tensor([], dtype=torch.int64, requires_grad=True)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_empty_sequence_accumulates_leaf_gradient(device):
    require_rng_device(device)
    with preserve_random_streams():
        value = torch.tensor([], dtype=torch.float32, device=device, requires_grad=True)
        value.sum().backward()
        assert value.grad is not None
        assert value.grad.device.type == device
        assert tuple(value.grad.shape) == (0,)
