"""Scalar nonzero follows PyTorch's distinct tuple and matrix shapes."""
import numpy as np
import pytest
import torch


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("value,dtype,nonzero", [
    (False, torch.bool, False), (True, torch.bool, True),
    (0, torch.int64, False), (-3, torch.int64, True),
    (0.0, torch.float32, False), (float("nan"), torch.float32, True),
])
@pytest.mark.parametrize("method", [False, True])
@pytest.mark.parametrize("as_tuple", [False, True])
def test_nonzero_scalar(device, value, dtype, nonzero, method, as_tuple):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    value = torch.tensor(value, dtype=dtype, device=device)
    result = value.nonzero(as_tuple=as_tuple) if method else torch.nonzero(value, as_tuple=as_tuple)
    if as_tuple:
        assert isinstance(result, tuple) and len(result) == 1
        assert result[0].device.type == device and result[0].dtype == torch.int64
        np.testing.assert_array_equal(result[0].cpu().numpy(), [0] if nonzero else [])
    else:
        assert result.device.type == device and result.dtype == torch.int64
        assert tuple(result.shape) == ((1, 0) if nonzero else (0, 0))
