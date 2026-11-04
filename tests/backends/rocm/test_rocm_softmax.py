"""HIP row softmax values, exceptional rows and portable gradient boundaries."""

import numpy as np
import pytest
import jittor as jt
from _helpers.rocm import native_rocm
from jittor._runtime.dispatch import select_kernel

pytestmark = pytest.mark.skipif(not jt.compiler.has_rocm, reason='ROCm build required')


def reference(raw, axis=-1, log=False):
    values = raw.astype('float64')
    with np.errstate(invalid='ignore', divide='ignore'):
        shifted = values - values.max(axis, keepdims=True)
        normalizer = np.exp(shifted).sum(axis, keepdims=True)
        return shifted - np.log(normalizer) if log else np.exp(shifted) / normalizer


@pytest.mark.parametrize('shape', [(3, 1), (2, 33), (4, 128), (2, 257),
                                  (4, 2048), (2, 4096), (2, 3, 65)])
def test_inference_matches_wide_reference_and_preserves_input(shape):
    raw = np.random.RandomState(103).normal(size=shape).astype('float32') * 7
    raw[..., ::7] = -np.inf
    # A width-one row must contain a finite value for a finite reference.
    if shape[-1] == 1:
        raw.fill(2)
    with jt.no_grad():
        x = jt.array(raw)
        assert select_kernel('nn.softmax', x, dim=-1) is not None
        y = jt.nn.softmax(x, dim=-1)
        y.sync()
        assert y.location() == 'device' and str(y.dtype) == 'float32'
        assert not y.requires_grad
        np.testing.assert_allclose(y.numpy(), reference(raw), rtol=2e-5, atol=2e-6)
        np.testing.assert_array_equal(x.numpy(), raw)


def test_nonfinite_rows_and_extreme_values_follow_softmax_semantics():
    raw = np.array([[np.inf, 0, 1], [np.nan, 0, 1], [-np.inf]*3,
                    [10000, -10000, 0], [1, 1, 1]], dtype='float32')
    with jt.no_grad():
        y = jt.nn.softmax(jt.array(raw), -1)
        y.sync()
        assert y.location() == 'device'
        np.testing.assert_allclose(y.numpy(), reference(raw), rtol=2e-5, atol=2e-6, equal_nan=True)


def test_noncontiguous_input_is_read_without_mutating_its_base():
    raw = np.random.RandomState(104).normal(size=(3, 65, 2)).astype('float32')
    with jt.no_grad():
        base = jt.array(raw)
        y = jt.nn.softmax(base.transpose((0, 2, 1)), -1)
        y.sync()
        assert y.location() == 'device'
        np.testing.assert_allclose(y.numpy(), reference(raw.transpose(0, 2, 1)), rtol=2e-5, atol=2e-6)
        np.testing.assert_array_equal(base.numpy(), raw)


def test_training_declines_and_preserves_nonuniform_gradient():
    raw = np.random.RandomState(105).normal(size=(3, 65)).astype('float32')
    seed = np.random.RandomState(106).normal(size=raw.shape).astype('float32')
    x = jt.array(raw)
    assert select_kernel('nn.softmax', x, dim=-1) is None
    y = jt.nn.softmax(x, -1)
    dx = jt.grad((y * jt.array(seed)).sum(), x)
    expected = reference(raw)
    for actual, wanted in [(y, expected), (dx, expected * (seed - (seed * expected).sum(-1, keepdims=True)))]:
        actual.sync()
        assert actual.location() == 'device'
        np.testing.assert_allclose(actual.numpy(), wanted, rtol=3e-5, atol=3e-6)


def test_other_axes_and_log_softmax_retain_portable_values():
    raw = np.random.RandomState(107).normal(size=(4, 65)).astype('float32')
    with jt.no_grad():
        x = jt.array(raw)
        for dim, log in [(0, False), (-1, True)]:
            assert select_kernel('nn.softmax', x, dim=dim, log=log) is None
            y = jt.nn.softmax(x, dim=dim, log=log)
            y.sync()
            assert y.location() == 'device'
            np.testing.assert_allclose(y.numpy(), reference(raw, dim, log), rtol=2e-5, atol=2e-6)
        assert select_kernel('nn.softmax', x, dim=-1, zero_all_neg_inf=True) is None
        assert select_kernel('nn.softmax', x.half(), dim=-1) is None
        with jt.flag_scope(amp_reg=18):
            assert select_kernel('nn.softmax', x, dim=-1) is None


def test_empty_leading_dimension():
    with jt.no_grad():
        y = jt.nn.softmax(jt.empty((0, 128)), -1)
        y.sync()
        assert tuple(y.shape) == (0, 128)
