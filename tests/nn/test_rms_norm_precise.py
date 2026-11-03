"""Weighted RMS dtype, tail reduction, affine gradient and higher-AD contracts."""

import os

import jittor as jt
import numpy as np
import pytest

from jittor._runtime import forbid_backend_fallbacks
from jittor.backends.cuda.kernels.nn.rms_norm_precise_cuda import _rms_norm_precise_cuda


def _device():
    return os.environ.get("JITTOR_TEST_DEVICE", "cpu")


def _round(value, dtype):
    if dtype == "float16":
        return np.asarray(value, dtype=np.float16).astype(np.float64)
    if dtype == "bfloat16":
        bits = np.asarray(value, dtype=np.float32).view(np.uint32)
        bits = (bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) & np.uint32(0xffff0000)
        return bits.view(np.float32).astype(np.float64)
    return np.asarray(value, dtype=dtype).astype(np.float64)


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32", "float64"])
@pytest.mark.parametrize("shape", [(2, 3), (3, 129), (7, 1024)])
def test_weighted_rms_tail_dtype_and_complete_input_gradient(dtype, shape):
    rng = np.random.RandomState(392)
    source = _round(rng.normal(size=shape), dtype)
    weight_dtype = "float64" if dtype == "float64" else "float32"
    weights = _round(rng.uniform(.5, 1.5, size=shape[-1]), weight_dtype)
    seed = _round(rng.normal(size=shape), weight_dtype)
    inverse = 1 / np.sqrt(np.mean(source * source, axis=-1, keepdims=True) + 1e-6)
    expected = _round(source * inverse, dtype) * weights
    propagated = _round(seed * weights, dtype)
    dot = np.sum(propagated * source, axis=-1, keepdims=True)
    gradient = _round(propagated * inverse - source * dot * inverse**3 / shape[-1], dtype)
    with jt.flag_scope(use_cuda=int(_device() == "cuda"), amp_reg=0,
                       auto_convert_64_to_32=0), forbid_backend_fallbacks():
        x = jt.array(source, dtype=dtype)
        weight = jt.array(weights, dtype=weight_dtype).stop_grad()
        output = jt.nn.rms_norm(x, weight, 1e-6)
        dx = jt.grad((output * jt.array(seed, dtype=weight_dtype)).sum(), x)
        assert str(output.dtype) == weight_dtype
        assert str(dx.dtype) == dtype
        for value in (output, dx):
            value.sync()
            assert value.location() == ("device" if _device() == "cuda" else "cpu")
        # The narrow budgets retain the established native RMS normalization
        # contract; full-model admission uses independent Torch's stricter one.
        tolerance = {"float16": .004, "bfloat16": .03,
                     "float32": 1e-5, "float64": 2e-12}[dtype]
        np.testing.assert_allclose(output.numpy(), expected, atol=tolerance, rtol=tolerance)
        np.testing.assert_allclose(dx.float64().numpy(), gradient,
                                   atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_trainable_affine_weight_keeps_its_gradient(dtype):
    rng = np.random.RandomState(32)
    source = rng.normal(size=(3, 7)).astype(dtype)
    weights = rng.uniform(.5, 1.5, size=7).astype(dtype)
    seed = rng.normal(size=source.shape).astype(dtype)
    normalized = source / np.sqrt(np.mean(source * source, -1, keepdims=True) + 1e-6)
    with jt.flag_scope(use_cuda=int(_device() == "cuda"), amp_reg=0,
                       auto_convert_64_to_32=0), forbid_backend_fallbacks():
        x, weight = jt.array(source, dtype=dtype), jt.array(weights, dtype=dtype)
        assert _rms_norm_precise_cuda(x, weight, 1e-6) is None
        output = jt.nn.rms_norm(x, weight, 1e-6)
        dw = jt.grad((output * jt.array(seed, dtype=dtype)).sum(), weight)
        tolerance = 1e-5 if dtype == "float32" else 2e-12
        np.testing.assert_allclose(dw.numpy(), (seed * normalized).sum(0),
                                   atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_frozen_affine_path_keeps_analytic_second_derivatives(dtype):
    rng = np.random.RandomState(54)
    source = rng.normal(size=(2, 4)).astype(dtype)
    weights = rng.uniform(.5, 1.5, size=4).astype(dtype)
    seed = rng.normal(size=source.shape).astype(dtype)
    direction = rng.normal(size=source.shape).astype(dtype)
    x64, a, vector = source.astype("float64"), (seed * weights).astype("float64"), direction.astype("float64")
    inverse = 1 / np.sqrt(np.mean(x64 * x64, -1, keepdims=True) + 1e-6)
    xv = np.sum(x64 * vector, -1, keepdims=True)
    ax = np.sum(a * x64, -1, keepdims=True)
    av = np.sum(a * vector, -1, keepdims=True)
    expected = (-a * inverse**3 * xv / 4 - vector * ax * inverse**3 / 4
                - x64 * av * inverse**3 / 4 + 3 * x64 * ax * xv * inverse**5 / 16)
    with jt.flag_scope(use_cuda=int(_device() == "cuda"), amp_reg=0,
                       auto_convert_64_to_32=0), forbid_backend_fallbacks():
        x = jt.array(source, dtype=dtype)
        weight = jt.array(weights, dtype=dtype).stop_grad()
        output = jt.nn.rms_norm(x, weight, 1e-6)
        first = jt.grad((output * jt.array(seed, dtype=dtype)).sum(), x)
        second = jt.grad((first * jt.array(direction, dtype=dtype)).sum(), x)
        tolerance = 1e-5 if dtype == "float32" else 3e-12
        np.testing.assert_allclose(second.numpy(), expected, atol=tolerance, rtol=tolerance)


def test_optional_precision_kernel_declines_unsupported_layout_and_dtype():
    with jt.flag_scope(use_cuda=int(_device() == "cuda"), amp_reg=0,
                       auto_convert_64_to_32=0):
        x = jt.ones((2, 7), dtype="float32")
        weight = jt.ones((7,), dtype="float32").stop_grad()
        double = jt.ones((2, 7), dtype="float64")
        assert _rms_norm_precise_cuda(double, weight, 1e-6) is None
        with jt.flag_scope(transpose_storage_view=1):
            view = x.transpose()
        assert not view._storage_is_contiguous()
        assert _rms_norm_precise_cuda(view, jt.ones(2).stop_grad(), 1e-6) is None
        assert _rms_norm_precise_cuda(x, weight, -1.) is None


def test_residual_fanout_preserves_single_square_and_low_precision_rounding():
    if _device() != "cuda":
        pytest.skip("CUDA ordered RMS arithmetic")
    # Independent PyTorch CUDA eight-element fixture. All inputs are generic
    # FP16/FP32 values; combining the complete RMS VJP before its residual
    # rounds coordinate 3 to 2.7265625 instead of 2.724609375.
    hidden = [-.245849609375, -.436279296875, -1.625, .379150390625,
              1.1328125, -.64208984375, -.1370849609375, .2464599609375]
    residual = [-.8846797347068787, .31875911355018616, .4967401325702667,
                -.9920704364776611, 2.3839356899261475, -1.1281813383102417,
                -.7447981238365173, .11252710223197937]
    seed = [.2954454720020294, .5590096116065979, 1.262969732284546,
            1.6673214435577393, .581255316734314, -.23772980272769928,
            .4704267978668213, .32484787702560425]
    expected = np.array([[.4599609375, .916015625, 2.048828125, 2.724609375,
                          1.0341796875, -.4306640625, .7529296875, .5419921875]],
                        dtype="float32")
    outputs = {}
    for order in (0, 1):
        with jt.flag_scope(use_cuda=1, amp_reg=0, gradient_accumulation_order=order), \
                forbid_backend_fallbacks():
            h = jt.array(np.array([hidden], dtype="float32")).float16()
            value = h + jt.array(np.array([residual], dtype="float32")).stop_grad()
            weights = jt.ones((8,)).stop_grad()
            output = value + jt.nn.rms_norm(value, weights, 1e-6)
            gradient = jt.grad((output * jt.array(np.array([seed], dtype="float32"))).sum(), h)
            outputs[order] = gradient.float16().float32().numpy()
    np.testing.assert_array_equal(outputs[1], expected)
    assert outputs[0][0, 3] == 2.7265625
    assert np.count_nonzero(outputs[0] != expected) == 1
