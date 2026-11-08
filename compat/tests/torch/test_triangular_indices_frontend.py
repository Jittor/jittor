"""Triangular index factories keep row-major order, widths and placement."""

import numpy as np
import pytest
import torch


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("name", ["tril", "triu"])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("shape", [(0, 3), (3, 0), (1, 1), (2, 3), (5, 2)])
@pytest.mark.parametrize("offset", [-7, -1, 0, 1, 7])
def test_triangular_indices_values_order_and_width(device, name, dtype, shape, offset):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    got = getattr(torch, name + "_indices")(*shape, offset=offset, dtype=dtype, device=device)
    expected = np.stack(getattr(np, name + "_indices")(shape[0], k=offset, m=shape[1]), 0)
    assert got.device.type == device and got.dtype == dtype
    assert not got.requires_grad
    if hasattr(torch, "_torch_compat_install_context"):
        got.sync()
        assert got.placement_backend == (1 if device == "cuda" else 0)
    np.testing.assert_array_equal(got.cpu().numpy(), expected)


@pytest.mark.parametrize("name", ["tril", "triu"])
def test_triangular_indices_validate_shape_and_dtype(name):
    operation = getattr(torch, name + "_indices")
    assert operation(2, 3).dtype == torch.int64
    with pytest.raises((RuntimeError, ValueError)):
        operation(-1, 3)
    with pytest.raises((RuntimeError, ValueError)):
        operation(2, 3, dtype=torch.float32)
