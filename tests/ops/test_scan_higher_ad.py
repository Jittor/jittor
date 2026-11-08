"""Native cumsum higher derivatives against real-inner-product formulas."""

import jittor as jt
import numpy as np
import pytest

from _helpers import capability


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda":
        capability.require_accelerator("cuda")
    with jt.flag_scope(use_cuda=int(request.param == "cuda")):
        yield request.param


def _data_and_directions(shape, dtype):
    rng = np.random.RandomState(926)
    arrays = []
    for _ in range(3):
        value = 0.1 + 0.2 * rng.normal(size=shape)
        if np.dtype(dtype).kind == "c":
            value = value + 0.2j * rng.normal(size=shape)
        arrays.append(value.astype(dtype))
    return arrays


def _transpose_scan(value, axis):
    """Adjoint of inclusive cumsum, also for the complex real inner product."""
    return np.flip(np.cumsum(np.flip(value, axis=axis), axis=axis), axis=axis)


def _real_inner_product(left, right):
    if left.dtype.is_complex():
        return (left * right.conj()).real.sum()
    return (left * right).sum()


def _assert_values(device, dtype, actual, expected):
    # Check physical placement before any numpy() call can fetch to the host.
    for value, reference in zip(actual, expected):
        value.sync()
        assert value.location() == ("device" if device == "cuda" else "cpu")
        assert str(value.dtype) == dtype
        assert tuple(value.shape) == reference.shape
    tolerance = 5e-5 if dtype in ("float32", "complex64") else 2e-11
    for value, reference in zip(actual, expected):
        np.testing.assert_allclose(
            value.numpy(), reference, rtol=tolerance, atol=tolerance
        )


@pytest.mark.parametrize("dtype", ["float32", "float64", "complex64", "complex128"])
@pytest.mark.parametrize(
    "shape, axis",
    [((7,), 0), ((3, 5), -1), ((2, 3, 4), -2)],
    ids=["vector", "multirow-last", "nonlast-axis"],
)
def test_cumsum_quartic_loss_hvp(device, dtype, shape, axis):
    """For L=sum(|Sx|^4), differentiate Re(<grad L, v>) once more.

    Complex directions contain both real and imaginary components. The oracle
    is the real Hessian action, not a holomorphic second derivative:
    H v = S* [4 |Sx|^2 Sv + 8 Re(conj(Sx) Sv) Sx].
    """
    data, direction, _ = _data_and_directions(shape, dtype)
    x = jt.array(data, dtype=dtype)
    v = jt.array(direction, dtype=dtype).stop_grad()
    value = jt.cumsum(x, axis)
    magnitude = (value * value.conj()).real if np.iscomplexobj(data) else value * value
    loss = (magnitude * magnitude).sum()
    gradient = jt.grad(loss, x, retain_graph=True)
    hvp = jt.grad(_real_inner_product(gradient, v), x, retain_graph=True)

    expected_value = np.cumsum(data, axis=axis)
    scanned_direction = np.cumsum(direction, axis=axis)
    magnitude = (expected_value * expected_value.conj()).real
    mixed = (expected_value.conj() * scanned_direction).real
    expected_gradient = _transpose_scan(4 * magnitude * expected_value, axis)
    expected_hvp = _transpose_scan(
        4 * magnitude * scanned_direction + 8 * mixed * expected_value, axis
    )
    _assert_values(
        device,
        dtype,
        (value, gradient, hvp),
        (expected_value, expected_gradient, expected_hvp),
    )


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_cumsum_quartic_loss_third_derivative(device, dtype):
    """A nonzero third derivative requires another differentiable scan VJP."""
    shape, axis = (2, 3, 4), 1
    data, direction, second_direction = _data_and_directions(shape, dtype)
    x = jt.array(data, dtype=dtype)
    v = jt.array(direction, dtype=dtype).stop_grad()
    w = jt.array(second_direction, dtype=dtype).stop_grad()
    value = jt.cumsum(x, axis)
    squared = value * value
    gradient = jt.grad((squared * squared).sum(), x, retain_graph=True)
    hvp = jt.grad((gradient * v).sum(), x, retain_graph=True)
    third = jt.grad((hvp * w).sum(), x, retain_graph=True)

    expected_value = np.cumsum(data, axis=axis)
    scanned_direction = np.cumsum(direction, axis=axis)
    scanned_second_direction = np.cumsum(second_direction, axis=axis)
    expected_gradient = _transpose_scan(4 * expected_value**3, axis)
    expected_hvp = _transpose_scan(12 * expected_value**2 * scanned_direction, axis)
    expected_third = _transpose_scan(
        24 * expected_value * scanned_direction * scanned_second_direction, axis
    )
    _assert_values(
        device,
        dtype,
        (value, gradient, hvp, third),
        (expected_value, expected_gradient, expected_hvp, expected_third),
    )
