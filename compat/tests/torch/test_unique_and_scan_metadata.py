import pytest
import torch
from _helpers.common import get_all_device_types


@pytest.mark.parametrize('device', get_all_device_types())
def test_unique_metadata_and_csr_pointer_are_int64(device):
    x = torch.tensor([[1, 0], [1, 0], [2, 0]], device=device)
    _, inverse, counts = torch.unique(x, dim=0, return_inverse=True, return_counts=True)
    assert inverse.dtype == torch.int64
    assert counts.dtype == torch.int64
    ptr = torch.cat([counts.new_zeros(1), counts.cumsum(0)])
    assert ptr.dtype == torch.int64
    assert ptr.cpu().tolist() == [0, 2, 3]


@pytest.mark.parametrize('device', get_all_device_types())
def test_integer_scans_promote_before_accumulating(device):
    x = torch.tensor([2_000_000_000, 2_000_000_000], dtype=torch.int32, device=device)
    y = torch.cumsum(x, dim=0)
    assert y.dtype == torch.int64
    assert y.cpu().tolist() == [2_000_000_000, 4_000_000_000]
    assert x.cumsum(0).cpu().tolist() == [2_000_000_000, 4_000_000_000]


@pytest.mark.parametrize('device', get_all_device_types())
def test_cumsum_explicit_dtype_and_out(device):
    x = torch.tensor([30000, 10000], dtype=torch.int32, device=device)
    y = torch.cumsum(x, 0, dtype=torch.int64)
    assert y.dtype == torch.int64 and y.cpu().tolist() == [30000, 40000]
    out = torch.empty(2, dtype=torch.int64, device=device)
    assert torch.cumsum(x, 0, out=out) is out
    assert out.cpu().tolist() == [30000, 40000]


@pytest.mark.parametrize('device', get_all_device_types())
def test_unique_consecutive_metadata_is_int64(device):
    x = torch.tensor([1, 1, 2, 3, 3], device=device)
    _, inverse, counts = torch.unique_consecutive(x, return_inverse=True, return_counts=True)
    assert inverse.dtype == torch.int64
    assert counts.dtype == torch.int64


@pytest.mark.parametrize('device', get_all_device_types())
@pytest.mark.parametrize('name', ['uint16', 'uint32', 'uint64'])
def test_unsigned_scans_promote_to_int64(device, name):
    x = torch.tensor([1, 2, 3], dtype=getattr(torch, name), device=device)
    result = x.cumsum(0)
    assert result.dtype == torch.int64
    assert result.device == x.device
    assert result.cpu().tolist() == [1, 3, 6]


@pytest.mark.parametrize('device', get_all_device_types())
@pytest.mark.parametrize('dtype,out_dtype', [
    (torch.int64, torch.int32), (torch.float64, torch.float32)])
def test_cumsum_rejects_conflicting_dtype_before_writing_out(device, dtype, out_dtype):
    x = torch.tensor([1, 2], dtype=torch.int32, device=device)
    out = torch.tensor([7, 8], dtype=out_dtype, device=device)
    with pytest.raises(RuntimeError):
        torch.cumsum(x, 0, dtype=dtype, out=out)
    assert out.dtype == out_dtype
    assert out.cpu().tolist() == [7, 8]
