"""Boolean reductions share native kernels while retaining frontend contracts."""
import numpy as np
import pytest
import torch
import jittor as jt

from _helpers.common import kernel_launches
from _rng_init_support import require_rng_device


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("operation", ["all", "any"])
@pytest.mark.parametrize("layout", ["dense", "transpose", "broadcast"])
def test_logical_reduction_preserves_frontend_device_and_views(device, operation, layout):
    require_rng_device(device)
    array = np.array([[True, False, True], [True, True, True]])
    value = torch.tensor(array, device=device)
    if layout == "transpose":
        value, array = value.t(), array.T
    elif layout == "broadcast":
        value = value[:1].expand(4, 3)
        array = np.broadcast_to(array[:1], (4, 3))
    reducer = getattr(np, operation)
    for dim in (None, 0, -1, (0, 1)):
        for keepdim in (False, True):
            result = getattr(value, operation)(dim=dim, keepdim=keepdim)
            result.sync()
            assert type(result) is torch.Tensor
            assert result.dtype == torch.bool and result.device.type == device
            assert result.location() == ("cpu" if device == "cpu" else "device")
            np.testing.assert_array_equal(result.cpu().numpy(), reducer(array, axis=dim, keepdims=keepdim))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", ["int64", "float32", "uint8"])
def test_numeric_truth_reduction_preserves_values_and_output_dtype(device, dtype):
    require_rng_device(device)
    values = [[0., -0., float("nan")], [2., -3., float("inf")]] if dtype == "float32" else [[0, 0, 2], [2, 4, 8]]
    array = np.array(values, dtype=dtype)
    value = torch.tensor(array, device=device)
    for operation in ("all", "any"):
        for dim in (None, -1):
            result = getattr(torch, operation)(value, dim=dim)
            result.sync()
            assert type(result) is torch.Tensor and result.device.type == device
            expected_dtype = torch.uint8 if dtype == "uint8" else torch.bool
            assert result.dtype == expected_dtype
            np.testing.assert_array_equal(result.cpu().numpy(), getattr(np, operation)(array, axis=dim))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_boolean_native_entry_accepts_frontend_and_builds_fresh_results(device):
    require_rng_device(device)
    value = torch.tensor([[True, True], [True, True]], device=device)
    first = jt.core._logical_reduce(value, (0, 1), False, True)
    assert first is not NotImplemented and type(first) is torch.Tensor
    first.sync()
    assert first.device.type == device and first.dtype == torch.bool and first.item()
    value.fill_(False)
    second = jt.core._logical_reduce(value, (0, 1), False, True)
    assert second is not first and type(second) is torch.Tensor
    assert not second.item() and first.item()
    # Unsupported operands still hand back to the general operation.
    assert jt.core._logical_reduce(object(), (), False, True) is NotImplemented
    integer = torch.tensor([0, 1], dtype=torch.int64, device=device)
    assert jt.core._logical_reduce(integer, (), False, True) is NotImplemented


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_empty_and_scalar_logical_reductions(device):
    require_rng_device(device)
    for array in (np.empty((2, 0), dtype=np.bool_), np.array(False), np.array(True)):
        value = torch.tensor(array, device=device)
        for operation in ("all", "any"):
            for dim in (None, ()):
                result = getattr(value, operation)(dim=dim)
                assert result.dtype == torch.bool and result.device.type == device
                np.testing.assert_array_equal(result.cpu().numpy(), getattr(np, operation)(array, axis=dim))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_parameter_subclass_reduction_returns_tensor(device):
    require_rng_device(device)

    class MaskParameter(torch.nn.Parameter):
        pass

    value = MaskParameter(torch.tensor([True, False], device=device), requires_grad=False)
    for operation, expected in (("all", False), ("any", True)):
        result = getattr(value, operation)()
        assert type(result) is torch.Tensor and result.dtype == torch.bool
        assert result.device.type == device and result.item() == expected
        assert not result.requires_grad


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_boolean_truth_test_uses_one_reduction_kernel(device):
    require_rng_device(device)
    value = torch.tensor(np.arange(240) % 7 == 0, device=device)
    value.sync()
    for operation, expected in (("all", False), ("any", True)):
        launches, result = kernel_launches(lambda: bool(getattr(value, operation)()))
        assert result == expected and launches == 1
