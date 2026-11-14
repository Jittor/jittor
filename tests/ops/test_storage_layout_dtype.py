"""Physical layout aliases preserve dtype promotion and gradient values."""

import sys

import numpy as np
import pytest

import jittor as jt
from jittor._runtime.fallback import forbid_backend_fallbacks
from _helpers import torch_runtime


@pytest.fixture(autouse=True)
def native_device_only():
    # What this file needs is that jittor's own runtime is under test, which is
    # what `torch_shim_loaded` answers. `'torch' not in sys.modules` was a
    # different claim: a native test may import an independent PyTorch to
    # compare against, and xdist shares a worker between files, so one such
    # import turned this whole file red in the smoke tier.
    assert not torch_runtime.torch_shim_loaded()
    assert 'jittor.compat.torch' not in sys.modules
    with forbid_backend_fallbacks():
        yield
        jt.sync_all(True)


def assert_device(value):
    value.sync()
    assert value.location() == ('device' if jt.flags.use_cuda else 'cpu')


@pytest.mark.parametrize('dtype', ['float16', 'bfloat16', 'float32', 'float64'])
@pytest.mark.parametrize('optimize', [0, 1])
@pytest.mark.parametrize('materialize', [False, True])
def test_transposed_sum_gradient_keeps_element_width(dtype, optimize, materialize):
    with jt.flag_scope(amp_reg=0, auto_convert_64_to_32=0,
                       propagate_storage_layout=optimize):
        root = jt.array(np.full((3, 8), 11.), dtype=dtype)
        view = (root * 3)._storage_permute((1, 0))
        if materialize:
            assert_device(view)
        gradient = jt.grad(view.sum(), root)
        assert_device(gradient)
        assert str(gradient.dtype) == dtype
        np.testing.assert_array_equal(gradient.float64().numpy(), np.full((3, 8), 3.))


@pytest.mark.parametrize('dtype', ['float16', 'bfloat16', 'float32', 'float64'])
@pytest.mark.parametrize('record_grad', [False, True])
@pytest.mark.parametrize('materialize_scalar', [False, True])
def test_layout_binary_preserves_broadcast_scalar_promotion(
        dtype, record_grad, materialize_scalar):
    with jt.flag_scope(amp_reg=0, auto_convert_64_to_32=0,
                       propagate_storage_layout=1, no_grad=not record_grad):
        root = jt.array(np.full((3, 8), 11.), dtype=dtype)
        view = root._storage_permute((1, 0))
        scalar = jt.array(3., dtype='float32').broadcast((8, 3))
        if materialize_scalar:
            assert_device(scalar)
        result = view * scalar
        assert_device(result)
        assert str(result.dtype) == dtype
        np.testing.assert_array_equal(result.float64().numpy(), np.full((8, 3), 33.))


@pytest.mark.parametrize('dtype', ['float16', 'bfloat16'])
@pytest.mark.parametrize('route', ['storage_permute', 'transpose'])
@pytest.mark.parametrize('record_grad', [False, True])
@pytest.mark.parametrize('materialize_scalar', [False, True])
def test_scalar_view_layout_matches_unoptimized_arithmetic(
        dtype, route, record_grad, materialize_scalar):
    values = np.arange(1, 25, dtype='float32').reshape(3, 8)
    outputs = []
    for optimize in (0, 1):
        with jt.flag_scope(amp_reg=0, auto_convert_64_to_32=0,
                           propagate_storage_layout=optimize,
                           no_grad=not record_grad):
            root = jt.array(values, dtype=dtype)
            scalar = jt.array(3., dtype='float32').broadcast((3, 8))
            if route == 'storage_permute':
                view = root._storage_permute((1, 0))
                scalar_view = scalar._storage_permute((1, 0))
            else:
                with jt.flag_scope(transpose_storage_view=1):
                    view = root.transpose(1, 0)
                    scalar_view = scalar.transpose(1, 0)
            if materialize_scalar:
                assert_device(scalar_view)
            result = view * scalar_view
            assert_device(result)
            assert bool(result.requires_grad) == record_grad
            if route == 'transpose':
                assert str(result.dtype) == 'float32'
            outputs.append(str(result.dtype))
            np.testing.assert_array_equal(result.float64().numpy(), (values * 3).T)
            if record_grad:
                gradient = jt.grad(result.sum(), root)
                assert_device(gradient)
                assert str(gradient.dtype) == dtype
                np.testing.assert_array_equal(gradient.float64().numpy(),
                                              np.full((3, 8), 3.))
    assert outputs[0] == outputs[1]


@pytest.mark.parametrize('kind', ['add', 'multiply', 'ternary'])
@pytest.mark.parametrize('record_grad', [False, True])
@pytest.mark.parametrize('materialize_view', [False, True])
@pytest.mark.parametrize('reverse', [False, True])
def test_finished_dense_and_permuted_operands_do_not_rewrite_recursively(
        kind, record_grad, materialize_view, reverse):
    with jt.flag_scope(propagate_storage_layout=1, transpose_storage_view=1,
                       no_grad=not record_grad):
        a_host = np.arange(1, 7, dtype='float32').reshape(3, 2)
        b_host = np.arange(7, 13, dtype='float32').reshape(2, 3)
        a, b = jt.array(a_host).sync(), jt.array(b_host).sync()
        view = b._storage_permute((1, 0))
        if materialize_view:
            assert_device(view)
        x, y = (view, a) if reverse else (a, view)
        if kind == 'add':
            result = x + y
            expected = a_host + b_host.T
            grad_a, grad_b = np.ones_like(a_host), np.ones_like(b_host)
        elif kind == 'multiply':
            result = x * y
            expected = a_host * b_host.T
            grad_a, grad_b = b_host.T, a_host.T
        else:
            mask_host = a_host % 2 == 0
            mask = jt.array(mask_host).sync()
            result = jt.ternary(mask, x, y)
            expected = np.where(mask_host, b_host.T if reverse else a_host,
                                a_host if reverse else b_host.T)
            grad_a = (~mask_host if reverse else mask_host).astype('float32')
            grad_b = (mask_host if reverse else ~mask_host).T.astype('float32')
        assert jt.flags.propagate_storage_layout == 1
        assert_device(result)
        np.testing.assert_array_equal(result.numpy(), expected)
        if record_grad:
            da, db = jt.grad(result.sum(), [a, b])
            assert_device(da)
            assert_device(db)
            np.testing.assert_array_equal(da.numpy(), grad_a)
            np.testing.assert_array_equal(db.numpy(), grad_b)
