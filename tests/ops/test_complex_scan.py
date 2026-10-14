"""Complex scans reuse real kernels without mixing nonfinite components."""

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


@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
@pytest.mark.parametrize("length", [5, 4099])
def test_complex_scan_values_and_gradient(device, dtype, length):
    rng = np.random.default_rng(925)
    data = (rng.normal(size=(2, length)) + 1j * rng.normal(size=(2, length))).astype(
        dtype
    )
    weights = (rng.normal(size=(2, length)) + 1j * rng.normal(size=(2, length))).astype(
        dtype
    )
    x, p = jt.array(data), jt.array(weights)
    value = jt.cumsum(x, 1)
    gradient = jt.grad((value * p.conj()).real.sum(), x)
    for tensor in (value, gradient):
        tensor.sync()
        assert tensor.location() == ("device" if device == "cuda" else "cpu")
        assert str(tensor.dtype) == dtype
    tolerance = 8e-5 if dtype == "complex64" else 2e-11
    np.testing.assert_allclose(
        value.numpy(), np.cumsum(data, axis=1), rtol=tolerance, atol=tolerance
    )
    expected = np.cumsum(weights[:, ::-1], axis=1)[:, ::-1]
    np.testing.assert_allclose(
        gradient.numpy(), expected, rtol=tolerance, atol=tolerance
    )


@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
@pytest.mark.parametrize("component", ["real", "imag"])
@pytest.mark.parametrize("special", [np.inf, np.nan])
def test_complex_scan_nonfinite_components(device, dtype, component, special):
    data = np.empty((2, 3), dtype=dtype)
    data.real = [[1, 2, 3], [4, 5, 6]]
    data.imag = 0.5
    getattr(data, component)[0, 0] = special
    expected = np.cumsum(data, axis=1)
    result = jt.cumsum(jt.array(data), 1)
    result.sync()
    assert result.location() == ("device" if device == "cuda" else "cpu")
    actual = result.numpy()
    np.testing.assert_array_equal(actual.real, expected.real)
    np.testing.assert_array_equal(actual.imag, expected.imag)


@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
def test_complex_scan_strided_input_and_nonlast_axis(device, dtype):
    data = np.arange(24).reshape(4, 6).astype(dtype)
    data.imag = data.real[::-1] / 8
    x = jt.array(data)
    sliced = x[:, ::2]
    sliced.sync()
    assert not sliced._storage_is_contiguous()
    result = jt.cumsum(sliced, 0)
    gradient = jt.grad(result.real.sum(), x)
    for tensor in (result, gradient):
        tensor.sync()
        assert tensor.location() == ("device" if device == "cuda" else "cpu")
    expected_gradient = np.zeros_like(data)
    expected_gradient[:, ::2] = np.arange(4, 0, -1)[:, None]
    np.testing.assert_array_equal(result.numpy(), np.cumsum(data[:, ::2], axis=0))
    np.testing.assert_array_equal(gradient.numpy(), expected_gradient)
