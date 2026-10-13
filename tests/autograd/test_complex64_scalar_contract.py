"""Scalar complex bridges and JVP regressions migrated from 32001665."""

import numpy as np
import pytest
import jittor as jt

from _helpers import capability


@pytest.fixture(params=["cpu", "cuda"])
def scalar_device(request):
    device = request.param
    if (
        device == "cuda"
        and not capability.check_accelerator("cuda", backend=jt).enabled
    ):
        pytest.skip("CUDA is unavailable in this build")
    with jt.flag_scope(use_cuda=int(device == "cuda")):
        yield device


def _check(value, expected, device):
    value.sync()
    assert value.location() == ("device" if device == "cuda" else "cpu")
    np.testing.assert_allclose(value.numpy(), expected, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
@pytest.mark.parametrize("use_reinterpret", [True, False])
def test_scalar_bridge_forward_and_direct_complex_double_backward(
    scalar_device, dtype, use_reinterpret, monkeypatch
):
    if not use_reinterpret:
        monkeypatch.setattr(jt, "reinterpret_view", None)
    z = jt.array(np.asarray(0.5 + 0.25j, dtype=dtype), dtype=dtype)
    roundtrip = jt.nn.view_as_complex(jt.nn.view_as_real(z))
    assert tuple(roundtrip.shape) == ()
    assert str(roundtrip.dtype) == dtype
    _check(roundtrip, 0.5 + 0.25j, scalar_device)
    # Build fresh inputs after the host read; residency assertions must not
    # mistake a prior numpy() fetch for an execution-device failure.
    z = jt.array(np.asarray(0.5 + 0.25j, dtype=dtype), dtype=dtype)
    gradient = jt.grad((z * z.conj()).real, z)
    second = jt.grad(gradient.real + gradient.imag, z)
    assert tuple(gradient.shape) == tuple(second.shape) == ()
    assert str(gradient.dtype) == str(second.dtype) == dtype
    second.sync()
    gradient.sync()
    _check(second, 2 + 2j, scalar_device)
    _check(gradient, 1 + 0.5j, scalar_device)


@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
@pytest.mark.parametrize("use_reinterpret", [True, False])
def test_scalar_bridge_composed_double_backward(
    scalar_device, dtype, use_reinterpret, monkeypatch
):
    if not use_reinterpret:
        monkeypatch.setattr(jt, "reinterpret_view", None)
    z = jt.array(np.asarray(0.5 + 0.25j, dtype=dtype), dtype=dtype)
    roundtrip = jt.nn.view_as_complex(jt.nn.view_as_real(z))
    gradient = jt.grad((roundtrip * roundtrip.conj()).real, z)
    second = jt.grad(gradient.real + gradient.imag, z)
    for value in (roundtrip, gradient, second):
        assert tuple(value.shape) == ()
        assert str(value.dtype) == dtype
        value.sync()
        assert value.location() == ("device" if scalar_device == "cuda" else "cpu")
    _check(roundtrip, 0.5 + 0.25j, scalar_device)
    _check(gradient, 1 + 0.5j, scalar_device)
    _check(second, 2 + 2j, scalar_device)


@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
@pytest.mark.parametrize("use_reinterpret", [True, False])
@pytest.mark.parametrize("direction", ["to_real", "to_complex"])
def test_scalar_bridge_composed_weighted_double_backward(
    scalar_device, dtype, use_reinterpret, direction, monkeypatch
):
    if not use_reinterpret:
        monkeypatch.setattr(jt, "reinterpret_view", None)
    real_dtype = "float32" if dtype == "complex64" else "float64"
    if direction == "to_real":
        x = jt.array(np.asarray(0.5 + 0.25j, dtype=dtype), dtype=dtype)
        pair = jt.nn.view_as_real(x)
        weights = jt.array([2.0, 3.0], dtype=real_dtype)
        loss = (pair * pair * weights).sum()
        gradient = jt.grad(loss, x)
        second = jt.grad(gradient.real + gradient.imag, x)
        expected = (2 + 1.5j, 4 + 6j)
    else:
        x = jt.array([0.5, 0.25], dtype=real_dtype)
        z = jt.nn.view_as_complex(x)
        real, imag = z.real, z.imag
        loss = 2 * real * real + 3 * imag * imag
        gradient = jt.grad(loss, x)
        second = jt.grad(gradient.sum(), x)
        expected = ([2.0, 1.5], [4.0, 6.0])
    # Unequal component weights expose a conjugation/sign error that a
    # symmetric roundtrip alone could hide.
    for value in (gradient, second):
        assert tuple(value.shape) == tuple(x.shape)
        assert value.dtype == x.dtype
        value.sync()
        assert value.location() == ("device" if scalar_device == "cuda" else "cpu")
    _check(gradient, expected[0], scalar_device)
    _check(second, expected[1], scalar_device)


def test_complex64_item_returns_python_complex(scalar_device):
    value = jt.array(np.asarray(0.5 + 0.25j, dtype=np.complex64))
    value.sync()
    assert value.location() == ("device" if scalar_device == "cuda" else "cpu")
    result = value.item()
    assert isinstance(result, complex)
    assert result == 0.5 + 0.25j


@pytest.mark.parametrize("create_graph", [False, True])
@pytest.mark.parametrize("real_input", [False, True])
def test_complex64_jvp_scalar_and_vector(scalar_device, create_graph, real_input):
    from jittor.autograd import jvp

    dtype = np.float32 if real_input else np.complex64
    data = np.asarray([0.5, -0.2], dtype=dtype)
    direction = np.asarray([0.3, 0.4], dtype=dtype)
    if not real_input:
        data += np.asarray([0.2j, -0.1j], dtype=dtype)
        direction += np.asarray([-0.1j, 0.2j], dtype=dtype)
    for scalar_output in (False, True):
        x, tangent = jt.array(data), jt.array(direction)
        coefficient = jt.array(np.asarray(1 + 0.5j, dtype=np.complex64))

        def function(value):
            result = value * value * coefficient
            return result.sum() if scalar_output else result

        out, derivative = jvp(function, x, tangent, create_graph=create_graph)
        expected_out = data**2 * (1 + 0.5j)
        expected_derivative = 2 * data * direction * (1 + 0.5j)
        if scalar_output:
            expected_out = expected_out.sum()
            expected_derivative = expected_derivative.sum()
        out.sync()
        derivative.sync()
        _check(out, expected_out, scalar_device)
        _check(derivative, expected_derivative, scalar_device)
