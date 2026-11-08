"""SiLU computes low-precision activations in FP32 and keeps native AD."""
import os

import jittor as jt
import numpy as np
import pytest

from jittor._runtime.fallback import forbid_backend_fallbacks


def _round(value, dtype):
    if dtype == "float16":
        return np.asarray(value, dtype=np.float16).astype(np.float64)
    bits = np.asarray(value, dtype=np.float32).view(np.uint32)
    rounded = (bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) & np.uint32(0xffff0000)
    return rounded.view(np.float32).astype(np.float64)


def _reference(x):
    sigmoid = 1. / (1. + np.exp(-x))
    first = sigmoid * (1. + x * (1. - sigmoid))
    second = sigmoid * (1. - sigmoid) * (2. + x * (1. - 2. * sigmoid))
    return x * sigmoid, first, second


def _device():
    device = os.environ.get("JITTOR_TEST_DEVICE", "cpu")
    assert device in ("cpu", "cuda")
    return device


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("autocast", [False, True])
def test_low_precision_silu_output_and_two_derivatives(dtype, autocast):
    device = _device()
    raw = np.array([-20., -10., -5., -3., -1., -.125, 0., .125, 1., 3., 5., 10., 20.])
    data = _round(raw, dtype)
    cotangent = _round(np.random.RandomState(7).uniform(.2, 1.1, raw.shape), dtype)
    direction = _round(np.random.RandomState(19).uniform(.2, 1.1, raw.shape), dtype)
    expected = _reference(data)
    preference = jt.amp_flags.prefer_bfloat16 if dtype == "bfloat16" else jt.amp_flags.prefer16
    amp_reg = preference if autocast else 0
    with jt.flag_scope(use_cuda=int(device == "cuda"), amp_reg=0), forbid_backend_fallbacks():
        x = jt.array(data.astype("float32")).cast(dtype)
        weight = jt.array(cotangent.astype("float32")).cast(dtype)
        vector = jt.array(direction.astype("float32")).cast(dtype)
        with jt.flag_scope(amp_reg=amp_reg):
            output = jt.nn.silu(x)
            assert jt.flags.amp_reg == amp_reg
        with jt.flag_scope(amp_reg=amp_reg):
            first = jt.grad((output * weight).sum(), x)
            second = jt.grad((first * vector).sum(), x)
            assert jt.flags.amp_reg == amp_reg
        references = (expected[0], expected[1] * cotangent, expected[2] * cotangent * direction)
        for index, (value, reference) in enumerate(zip((output, first, second), references)):
            assert str(value.dtype) == dtype
            value.sync()
            assert value.location() == ("device" if device == "cuda" else "cpu")
            observed = value.float32().numpy()
            assert np.isfinite(observed).all()
            reference = _round(reference, dtype)
            if index == 0:
                np.testing.assert_array_equal(observed, reference)
            else:
                # Preserve the frozen ecosystem backward budget, including
                # the public low-precision gradient boundary.
                error = np.max(np.abs(observed - reference)) / max(1e-8, np.max(np.abs(reference)))
                assert error <= 1e-2


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_wide_silu_preserves_analytic_second_derivatives(dtype):
    device = _device()
    raw = np.random.RandomState(23).uniform(-4., 4., (3, 5)).astype(dtype)
    cotangent = np.random.RandomState(7).uniform(.2, 1.1, raw.shape).astype(dtype)
    direction = np.random.RandomState(19).uniform(.2, 1.1, raw.shape).astype(dtype)
    expected = _reference(raw.astype("float64"))
    with jt.flag_scope(use_cuda=int(device == "cuda"), auto_convert_64_to_32=0, amp_reg=0), forbid_backend_fallbacks():
        x = jt.array(raw, dtype=dtype)
        sigmoid = x.sigmoid()
        sigmoid.sync()
        assert sigmoid.location() == ("device" if device == "cuda" else "cpu")
        tolerance = 2e-6 if dtype == "float32" else 2e-13
        np.testing.assert_allclose(sigmoid.numpy(), 1. / (1. + np.exp(-raw.astype("float64"))),
                                   rtol=tolerance, atol=tolerance)
        output = jt.nn.silu(x)
        first = jt.grad((output * jt.array(cotangent, dtype=dtype)).sum(), x)
        second = jt.grad((first * jt.array(direction, dtype=dtype)).sum(), x)
        references = (expected[0], expected[1] * cotangent, expected[2] * cotangent * direction)
        for value, reference in zip((output, first, second), references):
            assert str(value.dtype) == dtype
            value.sync()
            assert value.location() == ("device" if device == "cuda" else "cpu")
            observed = value.numpy()
            assert np.isfinite(observed).all()
            tolerance = 2e-6 if dtype == "float32" else 2e-13
            np.testing.assert_allclose(observed, reference, rtol=tolerance, atol=tolerance)
