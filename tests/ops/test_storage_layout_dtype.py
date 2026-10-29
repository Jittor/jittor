"""Physical layout aliases preserve dtype promotion and gradient values."""


import numpy as np
import pytest

import jittor as jt
from jittor._runtime.fallback import forbid_backend_fallbacks
from jittor.autograd.policy import NATIVE, get_policy


@pytest.fixture(autouse=True)
def native_device_only():
    assert get_policy() is NATIVE
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
