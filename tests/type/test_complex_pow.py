# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# Maintainers: Jittor Group
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
# Native complex `x ** y`. It had no kernel: every complex64/complex128 pow
# failed at materialisation with "Op 'pow' is not supported for dtype(s)".
#
# Forward references are numpy's (the principal branch, numpy and torch
# agree on every case here, zero bases included). The gradient is the
# conjugate-Wirtinger one of the other holomorphic ops: for
# loss = Re sum(W * x**y), d/dx = conj(W * y * x**(y-1)) and
# d/dy = conj(W * x**y * log x); a real exponent takes the real part. Where
# that is 0 * inf the derivative is 0, as torch defines it: d/dx at y == 0
# and d/dy at a zero base with a real exponent >= 0.
import warnings

import numpy as np
import pytest
import jittor as jt

from _helpers import capability


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if (
        request.param == "cuda"
        and not capability.check_accelerator("cuda", backend=jt).enabled
    ):
        pytest.skip("CUDA is unavailable in this build")
    with jt.flag_scope(use_cuda=int(request.param == "cuda")):
        yield request.param


_BASES = np.array([0, 1 + 1j, -2j, -1 + 0j, 0.3 - 2.5j, 1e-3 + 2j])
_EXPONENTS = [0, 1, 2, 3, 5, -1, -3, 0.5, 1.5, 1 + 0.25j, 0.5j, -0.5 + 1j]
_TOL = {"complex64": 2e-5, "complex128": 1e-12}


def _numpy_pow(x, y):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return x ** y


@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
@pytest.mark.parametrize("exponent", _EXPONENTS, ids=str)
def test_scalar_exponent_matches_numpy(device, dtype, exponent):
    x = _BASES.astype(dtype)
    out = (jt.array(x, dtype=dtype) ** exponent).numpy()
    assert out.dtype == np.dtype(dtype)
    np.testing.assert_allclose(out, _numpy_pow(x, exponent),
                               rtol=_TOL[dtype], atol=_TOL[dtype], equal_nan=True)


@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
def test_tensor_exponent_broadcasts(device, dtype):
    x = _BASES.astype(dtype)
    y = np.array([[2.0], [0.5 + 0.5j], [-1.0]], dtype=dtype)
    out = (jt.array(x, dtype=dtype) ** jt.array(y, dtype=dtype)).numpy()
    assert out.shape == (3, len(x))
    np.testing.assert_allclose(out, _numpy_pow(x, y),
                               rtol=_TOL[dtype], atol=_TOL[dtype], equal_nan=True)


def _expected_grads(x, y, w):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        z = x ** y
        gx = np.conj(w * y * x ** (y - 1))
        gy = np.conj(w * z * np.log(x))
    gx = np.where(y == 0, 0, gx)
    gy = np.where((x == 0) & (np.imag(y) == 0) & (np.real(y) >= 0), 0, gy)
    return gx, gy


@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
def test_gradients_of_base_and_exponent(device, dtype):
    x = np.array([0, 1 + 1j, -2j, -1 + 0.5j, 0.3 - 2.5j, 0, 0])
    y = np.array([2.0, 0.5, 1 + 0.25j, 3, 0.5j, 0, 1.5])
    w = np.array([0.7 - 0.2j, 1.1 + 0.4j, -0.3 + 0.9j, 0.5, -1.2 - 0.6j, 0.4j, 2.0])
    xv, yv = jt.array(x.astype(dtype), dtype=dtype), jt.array(y.astype(dtype), dtype=dtype)
    gx, gy = jt.grad((jt.array(w.astype(dtype)) * xv ** yv).real.sum(), [xv, yv])
    ex, ey = _expected_grads(x, y.astype(complex), w)
    tol = 1e-4 if dtype == "complex64" else 1e-10
    np.testing.assert_allclose(gx.numpy(), ex, rtol=tol, atol=tol)
    np.testing.assert_allclose(gy.numpy(), ey, rtol=tol, atol=tol)


@pytest.mark.parametrize("exponent", [2, 3, 0.5, -1, 1 + 0.25j], ids=str)
def test_gradient_with_a_python_scalar_exponent(device, exponent):
    x = np.array([1 + 1j, -2j, -1 + 0.5j, 0.3 - 2.5j])
    w = np.array([0.7 - 0.2j, 1.1 + 0.4j, -0.3 + 0.9j, 0.5])
    xv = jt.array(x, dtype="complex128")
    g = jt.grad((jt.array(w, dtype="complex128") * xv ** exponent).real.sum(), xv)
    ex, _ = _expected_grads(x, np.full(x.shape, exponent, dtype=complex), w)
    np.testing.assert_allclose(g.numpy(), ex, rtol=1e-10, atol=1e-10)


def test_real_exponent_gets_a_real_gradient(device):
    x = np.array([1 + 1j, -2j, -1 + 0.5j, 0.3 - 2.5j])
    y = np.array([2.0, 0.5, 3.0, -0.5])
    w = np.array([0.7 - 0.2j, 1.1 + 0.4j, -0.3 + 0.9j, 0.5])
    xv, yv = jt.array(x, dtype="complex128"), jt.array(y, dtype="float64")
    gx, gy = jt.grad((jt.array(w, dtype="complex128") * xv ** yv).real.sum(), [xv, yv])
    ex, ey = _expected_grads(x, y.astype(complex), w)
    assert gy.dtype == jt.float64
    np.testing.assert_allclose(gx.numpy(), ex, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(gy.numpy(), np.real(ey), rtol=1e-10, atol=1e-10)


def test_second_derivative(device):
    # Re(w x^3): first conj(3 w x^2); the gradient of its real part is conj(6 w x).
    x = np.array([1 + 1j, -2j, 0.3 - 2.5j])
    w = np.array([0.7 - 0.2j, 1.1 + 0.4j, -0.3 + 0.9j])
    xv = jt.array(x, dtype="complex128")
    exponent = jt.array(3.0, dtype="complex128")   # a tensor: no x*x*x shortcut
    first = jt.grad((jt.array(w, dtype="complex128") * xv ** exponent).real.sum(), xv)
    second = jt.grad(first.real.sum(), xv)
    np.testing.assert_allclose(second.numpy(), np.conj(6 * w * x), rtol=1e-10, atol=1e-10)
