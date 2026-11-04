"""Integral sum promotion, checked against independent integer arithmetic.

The expected dtype and overflow contracts were verified with genuine PyTorch.
CPU and CUDA tensors are explicitly placed rather than following global flags.
"""

import numpy as np
import pytest
import torch


@pytest.fixture(params=["cpu", pytest.param("cuda", marks=pytest.mark.cuda)])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    return request.param


INTEGER_DTYPES = (torch.bool, torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64)


@pytest.mark.parametrize("dtype", INTEGER_DTYPES)
def test_default_sum_promotes_integral_input(device, dtype):
    value = torch.tensor([[1, 0], [1, 1]], dtype=dtype, device=device)
    method = value.sum()
    module = torch.sum(value, dim=(0, 1), keepdim=True)
    assert method.dtype == module.dtype == torch.int64
    assert method.device.type == module.device.type == device
    assert tuple(method.shape) == ()
    assert tuple(module.shape) == (1, 1)
    assert method.item() == module.item() == 3


@pytest.mark.parametrize("dtype", INTEGER_DTYPES)
def test_explicit_sum_dtype_is_preserved(device, dtype):
    value = torch.tensor([[1, 0], [1, 1]], dtype=torch.int32, device=device)
    method = value.sum(dim=(0, 1), keepdim=True, dtype=dtype)
    module = torch.sum(value, dim=(0, 1), keepdim=True, dtype=dtype)
    assert method.dtype == module.dtype == dtype
    assert method.device.type == module.device.type == device
    assert tuple(method.shape) == tuple(module.shape) == (1, 1)
    expected = True if dtype == torch.bool else 3
    assert method.item() == module.item() == expected


def test_default_sum_accumulates_before_int32_overflow(device):
    value = torch.tensor([2147483647, 2147483647], dtype=torch.int32, device=device)
    assert value.sum().item() == 4294967294
    assert torch.sum(value, dtype=torch.int32).item() == -2


@pytest.mark.parametrize("dtype", (torch.int32, torch.int64))
def test_out_selects_dtype_and_preserves_identity(device, dtype):
    value = torch.tensor([[2147483647, 2147483647], [1, 2]], dtype=torch.int32, device=device)
    out = torch.empty((2, 1), dtype=dtype, device=device)
    result = torch.sum(value, dim=1, keepdim=True, out=out)
    assert result is out
    assert out.dtype == dtype
    assert out.device.type == device
    expected = [[-2 if dtype == torch.int32 else 4294967294], [3]]
    np.testing.assert_array_equal(out.detach().cpu().numpy(), expected)
    assert torch.sum(value, dim=1, keepdim=True, dtype=dtype, out=out) is out
    np.testing.assert_array_equal(out.detach().cpu().numpy(), expected)


def test_explicit_dtype_must_match_out(device):
    value = torch.ones((2, 2), dtype=torch.int32, device=device)
    out = torch.empty((2,), dtype=torch.int32, device=device)
    with pytest.raises(RuntimeError, match="dtype"):
        torch.sum(value, dim=1, dtype=torch.int64, out=out)
