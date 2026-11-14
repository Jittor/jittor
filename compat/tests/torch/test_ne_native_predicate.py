"""Inequality compares values without subtracting or materializing scalars."""
import numpy as np
import pytest
import torch

from _helpers.common import kernel_launches
from _rng_init_support import require_rng_device


def _check_spellings(value, other, expected, device):
    for operation in (lambda: value != other, lambda: value.ne(other),
                      lambda: torch.ne(value, other), lambda: torch.not_equal(value, other)):
        result = operation()
        result.sync()
        assert type(result) is torch.Tensor and result.dtype == torch.bool
        assert result.device.type == device
        assert result.location() == ("cpu" if device == "cpu" else "device")
        assert not result.requires_grad
        np.testing.assert_array_equal(result.cpu().numpy(), expected)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", ["int8", "int16", "int32", "int64"])
def test_inequality_at_signed_integer_boundaries(device, dtype):
    require_rng_device(device)
    limits = np.iinfo(dtype)
    array = np.array([limits.min, limits.min+1, -1, 0, 1, limits.max], dtype=dtype)
    value = torch.tensor(array, device=device)
    for other in (0, limits.min, limits.max):
        _check_spellings(value, other, array != other, device)
        operand = torch.tensor(np.array(other, dtype=dtype), device=device)
        _check_spellings(value, operand, array != other, device)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", ["float16", "float32", "float64"])
def test_nonfinite_values_aliases_and_signed_zero(device, dtype):
    require_rng_device(device)
    array = np.array([np.nan, np.inf, -np.inf, -0., 0., -2., 3.], dtype=dtype)
    value = torch.tensor(array, device=device, requires_grad=True)
    for other in (0., np.inf, -np.inf, np.nan):
        _check_spellings(value, other, array != other, device)
    for other in (value, value.view_as(value), value.clone()):
        _check_spellings(value, other, array != array, device)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("left,right", [("int32", "float32"), ("int64", "int32"),
                                       ("float16", "float32"), ("bool", "int32")])
def test_mixed_dtype_broadcast_comparisons(device, left, right):
    require_rng_device(device)
    a = np.array([[0], [1], [2]], dtype=left)
    b = np.array([[0, 1, 3]], dtype=right)
    value = torch.tensor(a, device=device)
    other = torch.tensor(b, device=device)
    _check_spellings(value, other, a != b, device)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_parameter_inequality_retains_tensor_ownership_and_fresh_values(device):
    require_rng_device(device)

    class LocalParameter(torch.nn.Parameter):
        pass

    value = LocalParameter(torch.tensor([-1., 0., 2.], device=device))
    first = value != 0
    first.sync()
    with torch.no_grad():
        value.fill_(0)
    _check_spellings(value, 0, np.array([False, False, False]), device)
    assert type(first) is torch.Tensor and first.device.type == device
    np.testing.assert_array_equal(first.cpu().numpy(), [True, False, True])


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_numeric_truth_comparison_uses_one_kernel(device):
    require_rng_device(device)
    value = torch.tensor(np.arange(-120, 120, dtype=np.int64), device=device)
    value.sync()
    launches, result = kernel_launches(lambda: value != 0)
    assert launches == 1
    np.testing.assert_array_equal(result.cpu().numpy(), np.arange(-120, 120) != 0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", ["int8", "uint8"])
def test_integer_scalar_conversion_wraps_in_the_operand_dtype(device, dtype):
    require_rng_device(device)
    limits = np.iinfo(dtype)
    array = np.array([limits.min, 0, limits.max], dtype=dtype)
    value = torch.tensor(array, device=device)
    for other in (-1, 128, 256, 2**40):
        wrapped = other % 256
        if dtype == "int8" and wrapped >= 128:
            wrapped -= 256
        _check_spellings(value, other, array != wrapped, device)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_floating_scalar_is_converted_without_losing_operand_precision(device):
    require_rng_device(device)
    half = torch.tensor([1.], dtype=torch.float16, device=device)
    _check_spellings(half, 1.00001, np.array([False]), device)
    exact = 1.0 + 2**-40
    double = torch.tensor([1., exact], dtype=torch.float64, device=device)
    _check_spellings(double, exact, np.array([True, False]), device)
