"""Mixed complex promotion must execute, not just advertise an output dtype."""

import operator

import numpy as np
import pytest
import jittor as jt

from _helpers import capability


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda":
        capability.require_accelerator("cuda")
    with jt.flag_scope(use_cuda=int(request.param == "cuda")):
        yield request.param


@pytest.mark.parametrize("other_dtype", ["float32", "float64", "complex128"])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("operation", ["add", "sub", "mul", "truediv", "eq", "ne"])
def test_mixed_complex_values(device, other_dtype, reverse, operation):
    left = np.array([1 + 2j, -3 + 0.5j], dtype=np.complex64)
    right = np.array([2 + 2**-40, 1 - 2**-39], dtype=other_dtype)
    if other_dtype.startswith("complex"):
        right += np.array([0.5j, -1.25j])
    if reverse:
        left, right = right, left
    function = getattr(operator, operation)
    result = function(
        jt.array(left, dtype=left.dtype.name), jt.array(right, dtype=right.dtype.name)
    )
    expected = function(left, right)
    result.sync()
    assert result.location() == ("device" if device == "cuda" else "cpu")
    assert str(result.dtype) == expected.dtype.name
    np.testing.assert_allclose(
        result.numpy(),
        expected,
        rtol=2e-7 if other_dtype == "float32" else 1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize("other_dtype", ["float64", "complex128"])
@pytest.mark.parametrize("reverse", [False, True])
def test_mixed_complex_gradient(device, other_dtype, reverse):
    left = np.array([1 + 2j, -3 + 0.5j], dtype=np.complex64)
    right = np.array([2 + 2**-40, 1 - 2**-39], dtype=other_dtype)
    if other_dtype.startswith("complex"):
        right += np.array([0.5j, -1.25j])
    x, y = jt.array(left), jt.array(right, dtype=other_dtype)
    result = (y * x if reverse else x * y).real.sum()
    gx, gy = jt.grad(result, [x, y])
    for value in (gx, gy):
        value.sync()
        assert value.location() == ("device" if device == "cuda" else "cpu")
    assert str(gx.dtype) == "complex64"
    assert str(gy.dtype) == other_dtype
    np.testing.assert_allclose(gx.numpy(), right.conj().astype(np.complex64), rtol=1e-6)
    expected_y = left.conj() if other_dtype.startswith("complex") else left.real
    np.testing.assert_allclose(gy.numpy(), expected_y, rtol=1e-12)


def test_mixed_comparison_keeps_double_precision(device):
    x = jt.array([1 + 0j, 1 + 0j, complex(float("nan"), 0)], dtype="complex64")
    y = jt.array([1, 1 + 2**-40, 1], dtype="float64")
    equal, unequal = x == y, y != x
    for value in (equal, unequal):
        value.sync()
        assert value.location() == ("device" if device == "cuda" else "cpu")
    np.testing.assert_array_equal(equal.numpy(), [True, False, False])
    np.testing.assert_array_equal(unequal.numpy(), [False, True, True])


def test_mixed_broadcast_and_second_gradient(device):
    a = np.array([[1 + 2j], [3 - 0.5j]], dtype=np.complex64)
    b = np.array([[0.5, -1, 2]], dtype=np.float64)
    x, y = jt.array(a), jt.array(b, dtype="float64")
    value = x * y
    loss = (value * value.conj()).real.sum()
    gx, gy = jt.grad(loss, [x, y])
    second = jt.grad(gx.real.sum(), y)
    for tensor in (value, gx, gy, second):
        tensor.sync()
        assert tensor.location() == ("device" if device == "cuda" else "cpu")
    np.testing.assert_allclose(value.numpy(), a * b, rtol=1e-12)
    np.testing.assert_allclose(gx.numpy(), 2 * a * np.sum(b * b), rtol=1e-6)
    np.testing.assert_allclose(gy.numpy(), 2 * b * np.sum(np.abs(a) ** 2), rtol=1e-6)
    np.testing.assert_allclose(second.numpy(), 4 * b * np.sum(a.real), rtol=1e-6)
