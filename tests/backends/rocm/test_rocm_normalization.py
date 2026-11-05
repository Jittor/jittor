"""HIP inference norms, numerical stability, layouts and gradient boundaries."""

import numpy as np
import pytest
import jittor as jt
from _helpers.rocm import native_rocm
from jittor._runtime.dispatch import select_kernel

pytestmark = pytest.mark.skipif(not jt.compiler.has_rocm, reason='ROCm build required')


def normalized(raw, kind, extent, eps=1e-5):
    wide = raw.astype('float64')
    shape = raw.shape
    if kind == 'group':
        wide = wide.reshape(shape[0], extent, -1)
        axes = (-1,)
    else:
        axes = tuple(range(wide.ndim - len(extent), wide.ndim))
    with np.errstate(invalid='ignore', divide='ignore'):
        centered = wide - wide.mean(axes, keepdims=True)
        inverse = 1 / np.sqrt((centered**2).mean(axes, keepdims=True) + eps)
    return (centered * inverse).reshape(shape), inverse


def affine_shape(raw, kind, extent):
    return (raw.shape[1],) if kind == 'group' else extent


def broadcast(value, raw, kind):
    return value.reshape((1, -1) + (1,) * (raw.ndim - 2)) if kind == 'group' else value


def public(kind):
    return jt.nn.group_norm if kind == 'group' else jt.nn.layer_norm


def operation(kind):
    return 'nn.group_norm' if kind == 'group' else 'nn.layer_norm.inference'


@pytest.mark.parametrize('kind,shape,extent', [
    ('group', (2, 6), 3), ('group', (2, 6, 7), 3),
    ('group', (2, 8, 5, 7), 4), ('group', (2, 8, 3, 5, 7), 4),
    ('group', (2, 8, 127, 129), 1),
    ('layer', (3, 65), (65,)), ('layer', (2, 3, 65), (65,)),
    ('layer', (2, 3, 5, 7), (5, 7)), ('layer', (2, 1), (1,)),
    ('layer', (2, 8193), (8193,)),
])
def test_public_inference_matches_wide_reference(kind, shape, extent):
    rng = np.random.RandomState(121)
    raw = rng.normal(size=shape).astype('float32')
    weights = rng.normal(size=affine_shape(raw, kind, extent)).astype('float32')
    biases = rng.normal(size=weights.shape).astype('float32')
    expected, _ = normalized(raw, kind, extent)
    expected = expected * broadcast(weights, raw, kind) + broadcast(biases, raw, kind)
    with jt.no_grad():
        x, w, b = map(jt.array, (raw, weights, biases))
        assert select_kernel(operation(kind), x, extent, w, b, 1e-5) is not None
        out = public(kind)(x, extent, w, b)
        out.sync()
        assert out.location() == 'device' and not out.requires_grad
        assert str(out.dtype) == 'float32'
        np.testing.assert_allclose(out.numpy(), expected, rtol=2e-5, atol=3e-6)
        for value, original in [(x, raw), (w, weights), (b, biases)]:
            np.testing.assert_array_equal(value.numpy(), original)


@pytest.mark.parametrize('kind', ['group', 'layer'])
@pytest.mark.parametrize('tensor_weight,tensor_bias', [(False, False), (True, False), (False, True)])
def test_scalar_and_optional_affine(kind, tensor_weight, tensor_bias):
    rng = np.random.RandomState(122)
    raw = rng.normal(size=(2, 6, 7)).astype('float32')
    extent = 3 if kind == 'group' else (7,)
    shape = affine_shape(raw, kind, extent)
    w = rng.normal(size=shape).astype('float32') if tensor_weight else 1.5
    b = rng.normal(size=shape).astype('float32') if tensor_bias else -0.2
    expected, _ = normalized(raw, kind, extent)
    expected = expected * (broadcast(w, raw, kind) if tensor_weight else w)
    expected += broadcast(b, raw, kind) if tensor_bias else b
    with jt.no_grad():
        out = public(kind)(jt.array(raw), extent,
                           jt.array(w) if tensor_weight else w,
                           jt.array(b) if tensor_bias else b)
        out.sync()
        assert out.location() == 'device'
        np.testing.assert_allclose(out.numpy(), expected, rtol=2e-5, atol=3e-6)


@pytest.mark.parametrize('kind', ['group', 'layer'])
def test_transpose_and_affine_slices_do_not_mutate_base(kind):
    rng = np.random.RandomState(123)
    raw = rng.normal(size=(2, 6, 5, 7)).astype('float32')
    extent = 3 if kind == 'group' else (7,)
    weights = rng.normal(size=2 * (6 if kind == 'group' else 7)).astype('float32')
    with jt.no_grad():
        base = jt.array(raw.transpose(0, 1, 3, 2).copy())
        wb, bb = jt.array(weights), jt.array(weights[::-1].copy())
        x, w, b = base.transpose((0, 1, 3, 2)), wb[::2], bb[::2]
        out = public(kind)(x, extent, w, b)
        out.sync()
        assert out.location() == 'device'
        expected, _ = normalized(raw, kind, extent)
        expected = expected * broadcast(weights[::2], raw, kind) + broadcast(weights[::-1][::2], raw, kind)
        np.testing.assert_allclose(out.numpy(), expected, rtol=2e-5, atol=3e-6)
        np.testing.assert_array_equal(base.numpy(), raw.transpose(0, 1, 3, 2))
        np.testing.assert_array_equal(wb.numpy(), weights)
        np.testing.assert_array_equal(bb.numpy(), weights[::-1])


@pytest.mark.parametrize('kind', ['group', 'layer'])
@pytest.mark.parametrize('with_affine', [False, True])
def test_large_offsets_overflow_and_nonfinite_rows(kind, with_affine):
    raw = np.array([[1e6, 1e6+.125, 1e6-.25, 1e6+.0625, 1e6],
                    [1e20, -1e20, 0, 2e20, -2e20],
                    [3e38, -3e38, 0, 2e38, -2e38],
                    [np.inf, 0, 1, 2, 3], [np.nan, 0, 1, 2, 3],
                    [3, 3, 3, 3, 3]], dtype='float32')
    if kind == 'group':
        raw = raw.reshape(6, 1, 5)
    extent = 1 if kind == 'group' else (5,)
    expected, _ = normalized(raw, kind, extent)
    shape = affine_shape(raw, kind, extent)
    weights = np.linspace(-.5, 1.5, np.prod(shape)).astype('float32').reshape(shape)
    biases = np.linspace(.1, .3, np.prod(shape)).astype('float32').reshape(shape)
    if with_affine:
        expected = expected * broadcast(weights, raw, kind) + broadcast(biases, raw, kind)
    with jt.no_grad():
        out = public(kind)(jt.array(raw), extent, jt.array(weights), jt.array(biases)) if with_affine else public(kind)(jt.array(raw), extent)
        out.sync()
        assert out.location() == 'device'
        np.testing.assert_allclose(out.numpy(), expected, rtol=3e-5, atol=3e-6, equal_nan=True)


@pytest.mark.parametrize('kind', ['group', 'layer'])
def test_training_declines_and_preserves_all_nonuniform_gradients(kind):
    rng = np.random.RandomState(124)
    raw = rng.normal(size=(2, 6, 5, 7)).astype('float32')
    extent = 3 if kind == 'group' else (5, 7)
    weights = rng.normal(size=affine_shape(raw, kind, extent)).astype('float32')
    biases = rng.normal(size=weights.shape).astype('float32')
    seed = rng.normal(size=raw.shape).astype('float32')
    x, w, b = map(jt.array, (raw, weights, biases))
    assert select_kernel(operation(kind), x, extent, w, b, 1e-5) is None
    out = public(kind)(x, extent, w, b)
    grads = jt.grad((out * jt.array(seed)).sum(), [x, w, b])
    xhat, inverse = normalized(raw, kind, extent)
    weighted = seed * broadcast(weights, raw, kind)
    if kind == 'group':
        h, g = xhat.reshape(2, extent, -1), weighted.reshape(2, extent, -1)
        dx = (inverse * (g - g.mean(-1, keepdims=True) - h * (g*h).mean(-1, keepdims=True))).reshape(raw.shape)
        reduce = (0, 2, 3)
    else:
        dx = inverse * (weighted - weighted.mean((-2, -1), keepdims=True)
                        - xhat * (weighted*xhat).mean((-2, -1), keepdims=True))
        reduce = (0, 1)
    wanted = [dx, (seed*xhat).sum(reduce), seed.sum(reduce)]
    for actual, expected in zip(grads, wanted):
        actual.sync()
        assert actual.location() == 'device'
        np.testing.assert_allclose(actual.numpy(), expected, rtol=4e-5, atol=2e-5)
    # A frozen activation alone does not permit inference if affine gradients
    # are needed. Every input participates in the provider's gradient guard.
    x.stop_grad()
    assert select_kernel(operation(kind), x, extent, w, b, 1e-5) is None
    w.stop_grad()
    assert select_kernel(operation(kind), x, extent, w, b, 1e-5) is None


def test_group_norm_silu_offer_and_fresh_affine_values():
    raw = np.random.RandomState(125).normal(size=(2, 8, 5, 7)).astype('float32')
    xhat, _ = normalized(raw, 'group', 4)
    with jt.no_grad():
        x, w, b = jt.array(raw), jt.ones(8), jt.zeros(8)
        raw_out = jt.nn.group_norm(x, 4, w, b)
        activated = jt.nn.silu(raw_out)
        activated.sync()
        assert activated.location() == 'device'
        np.testing.assert_allclose(activated.numpy(), xhat/(1+np.exp(-xhat)), rtol=2e-5, atol=3e-6)
        np.testing.assert_allclose(raw_out.numpy(), xhat, rtol=2e-5, atol=3e-6)
        w.assign(jt.full((8,), 2.0))
        b.assign(jt.full((8,), -0.5))
        changed = jt.nn.group_norm(x, 4, w, b)
        changed.sync()
        np.testing.assert_allclose(changed.numpy(), xhat*2-.5, rtol=2e-5, atol=3e-6)


def test_empty_shapes_and_precision_policy():
    from jittor.backends.rocm.kernels.normalization import supports_group_norm, supports_layer_norm
    with jt.no_grad():
        x, w, b = jt.ones((0, 8, 7)), jt.ones(8), jt.zeros(8)
        assert tuple(jt.nn.group_norm(x, 4, w, b).shape) == (0, 8, 7)
        assert tuple(jt.nn.layer_norm(x, (7,)).shape) == (0, 8, 7)
        assert not supports_group_norm(x.half(), 4, w.half(), b.half())
        assert not supports_group_norm(x, 4, w.float64(), b)
        for eps in [0, -1, float('nan'), float('inf')]:
            assert not supports_group_norm(x, 4, w, b, eps)
            assert not supports_layer_norm(x, (7,), 1, 0, eps)
        with jt.flag_scope(amp_reg=18):
            assert select_kernel('nn.group_norm', x, 4, w, b, 1e-5) is None
            assert select_kernel('nn.layer_norm.inference', x, (7,), 1, 0, 1e-5) is None
