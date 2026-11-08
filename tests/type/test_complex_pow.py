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
# Forward references are numpy's (the principal branch) away from a zero
# base, where numpy and torch agree; at a zero base jittor follows torch, see
# test_zero_base_follows_torch. The gradient is the
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


# Far from |z| = 1. The complex log, division, abs and sqrt formed
# re^2 + im^2, which overflows from |z| ~ 1e19 (complex64) / 1e154
# (complex128) and underflows below their reciprocals, although every result
# here is representable: z ** 0.5 came back NaN and z ** -1 zero or inf. Zero
# absolute tolerance, so a wrong zero cannot pass as a small number.
_EXTREMES = {"complex64": [1e20, 1e-30], "complex128": [1e200, 1e-200]}


@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
def test_far_from_unit_magnitude(device, dtype):
    for s in _EXTREMES[dtype]:
        z_np = np.array([s + s * 1j, -s + 0.5 * s * 1j, s * 1j], dtype=dtype)
        z = jt.array(z_np, dtype=dtype)
        # exp(w log z) carries log z's error times |w log z| (~460 at 1e200),
        # hence the comparison tolerances of the 2026-10-30 pow audit.
        rtol = 3e-5 if dtype == "complex64" else 1e-10
        for name, got, expected in (
            ("z ** 0.5", z ** 0.5, z_np ** 0.5),
            ("z ** -1", z ** -1, z_np ** -1),
            ("z ** (0.5+0.25j)", z ** (0.5 + 0.25j), z_np.astype(complex) ** (0.5 + 0.25j)),
            ("1 / z", 1 / z, 1 / z_np),
            ("abs(z)", z.abs(), np.abs(z_np)),
            ("sqrt(z)", z.sqrt(), np.sqrt(z_np)),
            ("log(z)", z.log(), np.log(z_np)),
        ):
            got = got.numpy()
            assert np.isfinite(got).all(), (name, s, got)
            np.testing.assert_allclose(got, np.asarray(expected).astype(got.dtype),
                                       rtol=rtol, atol=0, err_msg=f"{name} at {s}")


# A Python number is staged at single precision; against a complex128 operand
# it used to stay there, and z ** 1.2345678901234567 was off by about 2e-8.
@pytest.mark.parametrize("exponent", [1.2345678901234567, 1.2345678901234567 + 0.345678901234567j],
                         ids=["float", "complex"])
def test_python_exponent_keeps_double_precision_on_complex128(device, exponent):
    z_np = np.array([1 + 1j, -2j, 0.3 - 2.5j, 1.7 + 0.2j])
    z = jt.array(z_np, dtype="complex128")
    np.testing.assert_allclose((z ** exponent).numpy(), z_np ** exponent, rtol=1e-14, atol=0)
    # The same widening reaches every binary op, and a float64 base meeting a
    # complex scalar promotes to complex128 at full precision as well.
    np.testing.assert_allclose((z * exponent).numpy(), z_np * exponent, rtol=1e-15, atol=0)
    x_np = np.array([0.5, 1.25, 3.0])
    x = jt.array(x_np, dtype="float64")
    out = x + exponent
    assert str(out.dtype) == ("complex128" if isinstance(exponent, complex) else "float64")
    np.testing.assert_allclose(out.numpy(), x_np + exponent, rtol=1e-15, atol=0)


def test_python_exponent_stays_single_precision_on_complex64(device):
    # A Python number is a weak scalar: against complex64 the result is
    # complex64, computed at that precision.
    z = jt.array(np.array([1 + 1j, -2j], dtype="complex64"))
    assert str((z ** 1.2345678901234567).dtype) == "complex64"
    assert str((z * (0.5 + 0.25j)).dtype) == "complex64"


# A zero base follows torch, which differs from numpy there. A tensor exponent
# is std::pow, exp(y log 0): 0 for Re y > 0, inf+nanj for a real y < 0 and
# nan+nanj otherwise, 0 ** 0 included; its d/dx at y == 1 is 1 * 0 ** 0, NaN.
# torch special-cases the Python-scalar exponents 0 (ones, zero gradient) and
# 1 (the identity). Values recorded from torch 2.5.1 on CPU and CUDA.
def _classify(z):
    def part(v):
        return "nan" if np.isnan(v) else ("inf" if np.isposinf(v) else ("-inf" if np.isneginf(v) else float(v)))
    return [(part(c.real), part(c.imag)) for c in np.atleast_1d(z)]


_ZERO_TENSOR = {0: ("nan", "nan"), 1: (0.0, 0.0), 2: (0.0, 0.0), 0.5: (0.0, 0.0),
                -1: ("inf", "nan"), 0.5j: ("nan", "nan")}
_ZERO_SCALAR = {0: (1.0, 0.0), 1: (0.0, 0.0), 2: (0.0, 0.0), 0.5: (0.0, 0.0), -1: ("inf", "nan")}


@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
def test_zero_base_follows_torch(device, dtype):
    for y, expected in _ZERO_TENSOR.items():
        x = jt.array(np.zeros(1, dtype=dtype), dtype=dtype)
        got = (x ** jt.array(np.array([y], dtype=dtype), dtype=dtype)).numpy()
        assert _classify(got) == [expected], ("tensor", y, got)
    for y, expected in _ZERO_SCALAR.items():
        x = jt.array(np.zeros(1, dtype=dtype), dtype=dtype)
        got = (x ** y).numpy()
        assert _classify(got) == [expected], ("scalar", y, got)
    # Gradients of the base at 0.
    x = jt.array(np.array([0, 1 + 1j], dtype=dtype), dtype=dtype)
    g_scalar0 = jt.grad((x ** 0).real.sum(), x).numpy()
    g_scalar1 = jt.grad((x ** 1).real.sum(), x).numpy()
    one = jt.array(np.array([1.0, 1.0], dtype=dtype), dtype=dtype)
    g_tensor1 = jt.grad((x ** one).real.sum(), x).numpy()
    np.testing.assert_array_equal(g_scalar0, [0, 0])
    np.testing.assert_array_equal(g_scalar1, [1, 1])
    assert np.isnan(g_tensor1[0].real) and np.isnan(g_tensor1[0].imag)
    np.testing.assert_allclose(g_tensor1[1], 1)
