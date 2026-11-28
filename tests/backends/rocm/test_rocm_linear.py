"""Dense higher-rank hipBLAS products without forward reshape operators."""

import numpy as np
import pytest
import jittor as jt
from _helpers.rocm import native_rocm
from jittor._runtime.backend_libraries import get_library_ops
from jittor._runtime.dispatch import select_kernel

pytestmark = pytest.mark.skipif(not jt.compiler.has_rocm, reason='ROCm build required')




@pytest.mark.parametrize('prefix', [(2, 3), (2, 1, 3)])
@pytest.mark.parametrize('trans_b', [False, True])
def test_native_values_and_both_nonuniform_gradients(prefix, trans_b):
    rng = np.random.RandomState(101)
    raw = rng.normal(size=prefix + (4,)).astype('float32')
    weight = rng.normal(size=(4, 5)).astype('float32')
    seed = rng.normal(size=prefix + (5,)).astype('float32')
    stored = weight.T.copy() if trans_b else weight
    x, w = jt.array(raw), jt.array(stored)
    y = get_library_ops('hipblas').hipblas_matmul(x, w, False, trans_b)
    dx, dw = jt.grad((y * jt.array(seed)).sum(), [x, w])
    grad_w = raw.reshape(-1, 4).T @ seed.reshape(-1, 5)
    for out, expected in [(y, raw @ weight), (dx, seed @ weight.T),
                          (dw, grad_w.T if trans_b else grad_w)]:
        out.sync()
        assert out.location() == 'device'
        np.testing.assert_allclose(out.numpy(), expected, rtol=2e-5, atol=2e-5)
    np.testing.assert_array_equal(x.numpy(), raw)
    np.testing.assert_array_equal(w.numpy(), stored)


@pytest.mark.parametrize('strided', [False, True])
@pytest.mark.parametrize('transpose_weight', [False, True])
def test_public_dense_and_strided_input_paths(strided, transpose_weight):
    raw = np.arange(24, dtype='float32').reshape(2, 3, 4) / 8
    weight = np.arange(20, dtype='float32').reshape(4, 5) / 16
    with jt.no_grad():
        x = jt.array(raw.transpose(1, 0, 2).copy()).transpose((1, 0, 2)) if strided else jt.array(raw)
        w = jt.array(weight.T.copy()) if transpose_weight else jt.array(weight)
        out = jt.nn.matmul_transpose(x, w) if transpose_weight else jt.matmul(x, w)
        out.sync()
        assert out.location() == 'device'
        np.testing.assert_allclose(out.numpy(), raw @ weight, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize('trans_b', [False, True])
def test_weight_gradient_retains_second_derivative_edges(trans_b):
    raw = np.arange(24, dtype='float32').reshape(2, 3, 4) / 16
    weight = np.arange(20, dtype='float32').reshape(4, 5) / 16
    x = jt.array(raw)
    w = jt.array(weight.T.copy() if trans_b else weight)
    y = get_library_ops('hipblas').hipblas_matmul(x, w, False, trans_b)
    first = jt.grad(y.sqr().sum(), w)
    second = jt.grad(first.sum(), w)
    gram = raw.reshape(-1, 4).T @ raw.reshape(-1, 4)
    expected = 2 * gram @ np.ones((4, 5), dtype='float32')
    second.sync()
    assert second.location() == 'device'
    np.testing.assert_allclose(second.numpy(), expected.T if trans_b else expected, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize('shape', [(2, 0, 4), (2, 3, 0)])
def test_empty_rows_or_inner_dimension(shape):
    with jt.no_grad():
        raw, weight = np.zeros(shape, dtype='float32'), np.zeros((shape[-1], 5), dtype='float32')
        y = jt.matmul(jt.array(raw), jt.array(weight))
        y.sync()
        assert tuple(y.shape) == shape[:-1] + (5,)
        if y.numel():
            assert y.location() == 'device'
        np.testing.assert_array_equal(y.numpy(), raw @ weight)


@pytest.mark.parametrize("policy,dtype", [("prefer16", "float16"), ("prefer_bfloat16", "bfloat16")])
def test_higher_rank_amp_uses_low_precision_gemm(policy, dtype):
    with jt.no_grad():
        a, b = jt.ones((2, 3, 4)), jt.ones((4, 5))
        with jt.flag_scope(amp_reg=getattr(jt.amp_flags, policy)):
            assert select_kernel('matmul', a, b, False, False) is not None
            out = jt.matmul(a, b)
            out.sync()
            assert out.location() == "device" and str(out.dtype) == dtype
            np.testing.assert_array_equal(out.numpy(), np.full((2, 3, 5), 4))

@pytest.mark.parametrize("shape", [(5, 7), (2, 3, 7)])
def test_rocm_linear_bias_inference_and_gradient_boundary(shape):
    rng = np.random.RandomState(132)
    raw = rng.normal(size=shape).astype("float32")
    weights = rng.normal(size=(11, 7)).astype("float32")
    biases = rng.normal(size=11).astype("float32")
    with jt.no_grad():
        x, w, b = map(jt.array, (raw, weights, biases))
        assert select_kernel("linear_bias_accumulator", x, w, b) is not None
        result = jt.nn.linear(x, w, b)
        result.sync()
        assert result.location() == "device" and not result.requires_grad
        np.testing.assert_allclose(result.numpy(), raw @ weights.T + biases,
                                   rtol=3e-5, atol=3e-6)
    with jt.flag_scope(no_grad=0):
        w.start_grad()
        assert select_kernel("linear_bias_accumulator", x, w, b) is None


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_rocm_training_linear_bias_uses_single_rounding(dtype):
    rng = np.random.RandomState(199)
    x = jt.array(rng.normal(size=(2, 3, 9)).astype("float32")).cast(dtype)
    w = jt.array(rng.normal(size=(7, 9)).astype("float32")).cast(dtype)
    b = jt.array(rng.normal(size=7).astype("float32")).cast(dtype)
    assert select_kernel("linear_bias_accumulator", x, w, b) is not None
    xh, wh, bh = (v.numpy().astype("float32") for v in (x,w,b))
    expected = xh @ wh.T + bh
    if dtype == "float16":
        expected = expected.astype("float16").astype("float32")
    else:
        bits = expected.view(np.uint32)
        expected = ((bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000).view(np.float32)
    out = jt.nn.linear(x, w, b)
    out.sync()
    assert out.location() == "device" and str(out.dtype) == dtype
    np.testing.assert_allclose(out.numpy(), expected, rtol=0.002, atol=0.002)
    gradients = jt.grad(out.sum(), [x,w,b])
    seed = np.ones((2,3,7), dtype="float32")
    references = (seed @ wh, seed.reshape(-1,7).T @ xh.reshape(-1,9), seed.sum((0,1)))
    for g, reference in zip(gradients, references):
        g.sync()
        assert g.location() == "device" and np.isfinite(g.numpy()).all()
        np.testing.assert_allclose(g.numpy(), reference, rtol=0.008, atol=0.004)
