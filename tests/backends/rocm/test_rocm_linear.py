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


def test_higher_rank_amp_is_declined():
    with jt.no_grad():
        a, b = jt.ones((2, 3, 4)), jt.ones((4, 5))
        with jt.flag_scope(amp_reg=18):
            assert select_kernel('matmul', a, b, False, False) is None
