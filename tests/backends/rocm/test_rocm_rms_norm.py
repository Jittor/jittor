"""FP32 HIP RMSNorm inference, alias safety and training-path preservation."""

import numpy as np
import pytest
import jittor as jt
from _helpers.rocm import native_rocm
from jittor._runtime.dispatch import select_kernel

pytestmark = pytest.mark.skipif(not jt.compiler.has_rocm, reason='ROCm build required')


def reference(x, weight, epsilon):
    wide = x.astype('float64')
    return wide / np.sqrt(np.mean(wide**2, axis=-1, keepdims=True) + epsilon) * weight


@pytest.mark.parametrize('shape', [(1, 1, 1024), (4, 1, 1024), (4, 8, 128),
                                  (2, 512, 1024), (2, 33), (2, 1), (2, 8192)])
@pytest.mark.parametrize('epsilon', [1e-6, 0.01])
def test_public_inference_matches_wide_reference(shape, epsilon):
    rng = np.random.RandomState(87)
    raw = rng.normal(size=shape).astype('float32')
    weight = rng.normal(size=shape[-1]).astype('float32')
    with jt.no_grad():
        x, w = jt.array(raw), jt.array(weight)
        assert select_kernel('nn.rms_norm.inference', x, w, epsilon) is not None
        y = jt.nn.rms_norm(x, w, epsilon)
        y.sync()
        assert y.location() == 'device' and y.device_id >= 0
        assert str(y.dtype) == 'float32' and not y.requires_grad
        np.testing.assert_allclose(y.numpy(), reference(raw, weight, epsilon), rtol=2e-5, atol=2e-5)
        np.testing.assert_array_equal(x.numpy(), raw)
        np.testing.assert_array_equal(w.numpy(), weight)


def test_transpose_views_and_weight_slices_are_materialized_without_mutation():
    raw = np.random.RandomState(88).normal(size=(3, 65, 4)).astype('float32')
    weights = np.linspace(-2, 3, 130).astype('float32')
    with jt.no_grad():
        base, wb = jt.array(raw), jt.array(weights)
        x, w = base.transpose((0, 2, 1)), wb[::2]
        y = jt.nn.rms_norm(x, w, 1e-6)
        y.sync()
        assert y.location() == 'device'
        np.testing.assert_allclose(y.numpy(), reference(raw.transpose(0, 2, 1), weights[::2], 1e-6), rtol=2e-5, atol=2e-5)
        np.testing.assert_array_equal(base.numpy(), raw)
        np.testing.assert_array_equal(wb.numpy(), weights)


def test_training_declines_inference_and_preserves_both_nonuniform_gradients():
    rng = np.random.RandomState(89)
    raw = rng.normal(size=(2, 3, 65)).astype('float32')
    weight = rng.normal(size=65).astype('float32')
    seed = rng.normal(size=raw.shape).astype('float32')
    x, w = jt.array(raw), jt.array(weight)
    assert select_kernel('nn.rms_norm.inference', x, w, 1e-6) is None
    y = jt.nn.rms_norm(x, w, 1e-6)
    dx, dw = jt.grad((y * jt.array(seed)).sum(), [x, w])
    wide = raw.astype('float64')
    inv = 1 / np.sqrt((wide**2).mean(-1, keepdims=True) + 1e-6)
    weighted = seed * weight
    expected_dx = weighted * inv - wide * inv**3 * (weighted * wide).mean(-1, keepdims=True)
    expected_dw = (seed * wide * inv).sum((0, 1))
    for value, expected in [(y, reference(raw, weight, 1e-6)), (dx, expected_dx), (dw, expected_dw)]:
        value.sync()
        assert value.location() == 'device'
        np.testing.assert_allclose(value.numpy(), expected, rtol=3e-5, atol=3e-5)


def test_unsupported_dtype_amp_empty_and_epsilon_contracts():
    from jittor.backends.rocm.kernels.rms_norm import supported
    with jt.no_grad():
        x, w = jt.ones((0, 128)), jt.ones(128)
        y = jt.nn.rms_norm(x, w, 1e-6)
        assert tuple(y.shape) == (0, 128)
        assert not supported(x.half(), w.half())
        assert not supported(x, jt.ones(64))
        for eps in [0.0, -1.0, float('nan'), float('inf')]:
            assert not supported(x, w, eps)
        with jt.flag_scope(amp_reg=18):
            assert select_kernel('nn.rms_norm.inference', x, w, 1e-6) is None


def test_nonfinite_values_and_fp32_square_overflow_follow_opmath():
    raw = np.array([[0, 0, 0, 0, 0], [1e20]*5, [np.inf, 1, 2, 3, 4],
                    [np.nan, 1, 2, 3, 4]], dtype='float32')
    weight = np.linspace(-2, 2, 5).astype('float32')
    with np.errstate(over='ignore', invalid='ignore'):
        expected = raw / np.sqrt(np.mean(raw * raw, axis=-1, keepdims=True) + np.float32(1e-6)) * weight
    with jt.no_grad():
        y = jt.nn.rms_norm(jt.array(raw), jt.array(weight), 1e-6)
        y.sync()
        assert y.location() == 'device'
        np.testing.assert_allclose(y.numpy(), expected, rtol=2e-5, atol=2e-5, equal_nan=True)
