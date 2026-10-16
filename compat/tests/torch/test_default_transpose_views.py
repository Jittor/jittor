"""Default Torch permutations share storage and retain real native view history.

The same assertions run against independent Torch, without a view opt-in flag.
"""

import gc
import weakref

import numpy as np
import pytest
import torch

from _rng_init_support import _array, preserve_random_streams, require_rng_device


OPERATIONS = {
    'transpose': lambda x: x.transpose(0, 1),
    'torch_transpose': lambda x: torch.transpose(x, 0, 1),
    'permute': lambda x: x.permute(1, 0),
    'torch_permute': lambda x: torch.permute(x, (1, 0)),
    't': lambda x: x.t(),
    'torch_t': lambda x: torch.t(x),
    'T': lambda x: x.T,
    'mT': lambda x: x.mT,
    'identity_transpose': lambda x: x.transpose(0, 0),
    'identity_permute': lambda x: x.permute(0, 1),
}
VIEWS = ('transpose', 'permute', 't')


@pytest.fixture(autouse=True)
def no_fallback_or_native_default_change():
    if hasattr(torch, '_torch_compat_install_context'):
        import jittor as jt
        assert jt.flags.transpose_storage_view == 0
    with preserve_random_streams():
        yield
    if hasattr(torch, '_torch_compat_install_context'):
        assert jt.flags.transpose_storage_view == 0


def source(device, *, requires_grad=True):
    require_rng_device(device)
    value = torch.tensor([[1., 2., 3.], [4., 5., 6.]], device=device,
                         requires_grad=requires_grad)
    physical(value)
    return value


def physical(value):
    if hasattr(torch, '_torch_compat_install_context'):
        value.sync()
        if value.numel():
            assert value.location() == ('device' if value.device.type == 'cuda' else 'cpu')
    return value


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('operation', OPERATIONS)
@pytest.mark.parametrize('nonleaf', [False, True])
@pytest.mark.parametrize('no_grad', [False, True])
def test_default_permutation_has_real_storage_and_intermediate_gradient(
        device, operation, nonleaf, no_grad):
    root = source(device)
    x = root * 2 if nonleaf else root
    with torch.set_grad_enabled(not no_grad):
        y = OPERATIONS[operation](x)
    identity = operation.startswith('identity')
    assert y is not x and y._is_view() and y._base is x
    assert y.data_ptr() == x.data_ptr()
    assert tuple(y.shape) == ((2, 3) if identity else (3, 2))
    assert tuple(y.stride()) == ((3, 1) if identity else (1, 3))
    assert y.requires_grad and y.is_leaf == no_grad
    assert (y.grad_fn is None) == no_grad
    expected = _array(x) if identity else _array(x).T
    np.testing.assert_array_equal(_array(physical(y)), expected)
    weights = torch.arange(1, 7, device=device, dtype=torch.float32).reshape(y.shape)
    targets = (y, x, root) if nonleaf else (y, x)
    gradients = torch.autograd.grad((y * weights).sum(), targets, allow_unused=True)
    if no_grad:
        assert all(g is None for g in gradients)
    else:
        np.testing.assert_array_equal(_array(physical(gradients[0])), _array(weights))
        restored = _array(weights) if identity else _array(weights).T
        np.testing.assert_array_equal(_array(physical(gradients[1])), restored)
        if nonleaf:
            np.testing.assert_array_equal(_array(physical(gradients[2])), restored * 2)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('operation', VIEWS)
@pytest.mark.parametrize('target', ['source', 'alias'])
@pytest.mark.parametrize('self_copy', [False, True])
@pytest.mark.parametrize('no_grad_creation', [False, True])
def test_writeback_and_history_inspection_preserve_physical_view(
        device, operation, target, self_copy, no_grad_creation):
    x = source(device)
    with torch.set_grad_enabled(not no_grad_creation):
        y = OPERATIONS[operation](x)
    destination = x if target == 'source' else y
    with torch.no_grad():
        destination.copy_(destination) if self_copy else destination.fill_(7)
    expected = np.array([[1., 2., 3.], [4., 5., 6.]]) if self_copy else np.full((2, 3), 7.)
    np.testing.assert_array_equal(_array(physical(x)), expected)
    np.testing.assert_array_equal(_array(physical(y)), expected.T)
    assert y._base is x and y.data_ptr() == x.data_ptr()
    assert tuple(y.stride()) == (1, 3)
    if no_grad_creation:
        for inspect in (lambda: y.is_leaf, lambda: y.grad_fn, lambda: y * 2,
                        lambda: OPERATIONS[operation](y)):
            with pytest.raises(RuntimeError, match='no_grad mode'):
                inspect()
    else:
        assert not y.is_leaf and y.grad_fn is not None
        np.testing.assert_array_equal(_array(physical(y * 2)), expected.T * 2)
    assert tuple(y.stride()) == (1, 3)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('operation', VIEWS)
@pytest.mark.parametrize('reactivate_before_view', [False, True])
def test_frozen_source_is_not_reconnected_by_new_permutation(
        device, operation, reactivate_before_view):
    x = source(device)
    y = x.t()
    x.requires_grad_(False)
    with torch.no_grad():
        x.fill_(7)
    if reactivate_before_view:
        x.requires_grad_(True)
    z = OPERATIONS[operation](y)
    if not reactivate_before_view:
        x.requires_grad_(True)
    weights = torch.tensor([[1., 2., 3.], [4., 5., 6.]], device=device)
    gx, gy, gz = torch.autograd.grad((z * weights).sum(), (x, y, z), allow_unused=True)
    if reactivate_before_view:
        np.testing.assert_array_equal(_array(physical(gx)), _array(weights))
    else:
        assert gx is None
    np.testing.assert_array_equal(_array(physical(gy)), _array(weights).T)
    np.testing.assert_array_equal(_array(physical(gz)), _array(weights))


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('operation', VIEWS)
def test_actual_root_lifetime_and_data_allocation_are_separate(device, operation):
    x = source(device, requires_grad=False)
    y = OPERATIONS[operation](x)
    z = y.t()
    root = weakref.ref(x)
    assert z._base is x
    x.data = torch.full((2, 3), 9., device=device)
    assert y._base is x and z._base is x
    assert y._is_view() and y.data_ptr() != x.data_ptr()
    np.testing.assert_array_equal(_array(physical(y)), [[1., 4.], [2., 5.], [3., 6.]])
    del x, y
    gc.collect()
    assert root() is not None and z._base is root()
    np.testing.assert_array_equal(_array(physical(z)), [[1., 2., 3.], [4., 5., 6.]])
    del z
    gc.collect()
    assert root() is None


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('shape', [(), (3,), (2, 1, 3)])
def test_identity_and_unit_axis_permutations_keep_actual_ad_target(device, shape):
    require_rng_device(device)
    x = torch.ones(shape, device=device, requires_grad=True)
    y = x.permute(tuple(reversed(range(len(shape)))))
    assert y is not x and y._is_view() and y._base is x
    assert y.data_ptr() == x.data_ptr()
    assert not y.is_leaf and y.grad_fn is not None
    gy, gx = torch.autograd.grad((y * 3).sum(), (y, x))
    np.testing.assert_array_equal(_array(physical(gy)), np.full(tuple(y.shape), 3.))
    np.testing.assert_array_equal(_array(physical(gx)), np.full(shape, 3.))
    if len(shape) < 2:
        assert x.t()._base is x and x.transpose(0, 0)._base is x
    else:
        with pytest.raises(RuntimeError):
            x.t()


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_axis_errors_restore_default_and_named_axes_keep_views(device):
    x = source(device)
    for invalid, error in ((lambda: x.permute(0, 0), RuntimeError),
                           (lambda: x.permute(0), RuntimeError),
                           (lambda: x.transpose(0, 4), IndexError)):
        with pytest.raises(error):
            invalid()
        y = x.transpose(dim0=-1, dim1=-2)
        assert y._base is x and tuple(y.stride()) == (1, 3)
        gradient, = torch.autograd.grad((y * y).sum(), (x,))
        np.testing.assert_array_equal(_array(physical(gradient)), _array(x) * 2)
    assert x.permute(dims=(1, 0))._base is x


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('kind', ['transpose', 'holes', 'expand'])
@pytest.mark.parametrize('contiguous', [False, True])
def test_clone_preserves_only_nonoverlapping_dense_format(device, kind, contiguous):
    x = source(device)
    y = x.t() if kind == 'transpose' else x[:, ::2] if kind == 'holes' else x[:1].expand(4, 3)
    z = y.clone(memory_format=torch.contiguous_format if contiguous else torch.preserve_format)
    assert not z._is_view() and z._base is None
    assert z.data_ptr() != y.data_ptr()
    assert tuple(z.stride()) == ((1, 3) if kind == 'transpose' and not contiguous
                                 else (int(z.shape[1]), 1))
    np.testing.assert_array_equal(_array(physical(z)), _array(y))
    gradient, = torch.autograd.grad((z * z).sum(), (x,), create_graph=True)
    second, = torch.autograd.grad(gradient.sum(), (x,))
    expected = np.full((2, 3), 2.)
    if kind == 'holes':
        expected[:, 1] = 0
    elif kind == 'expand':
        expected[0] = 8
        expected[1] = 0
    np.testing.assert_array_equal(_array(physical(second)), expected)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32,
                                   torch.float64, torch.int64, torch.bool])
def test_view_dtype_and_clone_storage_are_not_just_metadata(device, dtype):
    require_rng_device(device)
    values = [[1, 0, 1], [0, 1, 0]]
    x = torch.tensor(values, dtype=dtype, device=device)
    y = x.permute(1, 0)
    copied = y.clone()
    assert y.dtype == dtype and copied.dtype == dtype
    assert y.data_ptr() == x.data_ptr() and copied.data_ptr() != x.data_ptr()
    assert tuple(y.stride()) == tuple(copied.stride()) == (1, 3)
    y.fill_(0)
    np.testing.assert_array_equal(_array(physical(x)), np.zeros((2, 3)))
    np.testing.assert_array_equal(_array(physical(copied)), np.array(values).T)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_empty_permutation_and_detach_keep_distinct_provenance(device):
    require_rng_device(device)
    x = torch.empty((0, 3), device=device, requires_grad=True)
    y = x.t()
    assert tuple(y.shape) == (3, 0) and tuple(y.stride()) == (1, 3)
    assert y._base is x and y._is_view() and y.grad_fn is not None
    gradient, = torch.autograd.grad(y.sum(), (x,))
    assert tuple(gradient.shape) == (0, 3)
    x = source(device)
    y = x.t().detach().requires_grad_(True)
    assert y._base is None and not y._is_view() and y.is_leaf
    assert y.data_ptr() == x.data_ptr()
    gx, gy = torch.autograd.grad((y * 3).sum(), (x, y), allow_unused=True)
    assert gx is None
    np.testing.assert_array_equal(_array(physical(gy)), np.full((3, 2), 3.))


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_composed_multiaxis_view_keeps_root_and_second_order_graph(device):
    require_rng_device(device)
    values = np.arange(1, 25, dtype='float32').reshape(2, 3, 4)
    x = torch.tensor(values, device=device, requires_grad=True)
    y = x.permute(2, 0, 1)
    z = y.transpose(0, 2)
    assert y._base is x and z._base is x
    assert tuple(y.stride()) == (1, 12, 4)
    assert tuple(z.stride()) == (4, 12, 1)
    assert x.data_ptr() == y.data_ptr() == z.data_ptr()
    np.testing.assert_array_equal(_array(physical(z)), values.transpose(1, 0, 2))
    weights = torch.arange(1, 25, device=device, dtype=torch.float32).reshape(z.shape)
    gradient, = torch.autograd.grad((z * z * weights).sum(), (x,), create_graph=True)
    expected_weights = _array(weights).transpose(1, 0, 2)
    np.testing.assert_array_equal(_array(physical(gradient)), 2 * values * expected_weights)
    second, = torch.autograd.grad(gradient.sum(), (x,))
    np.testing.assert_array_equal(_array(physical(second)), 2 * expected_weights)
    with torch.no_grad():
        z.fill_(9)
    np.testing.assert_array_equal(_array(physical(x)), np.full((2, 3, 4), 9.))


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('operation', ['unsqueeze', 'view', 'reshape'])
def test_viewable_noncontiguous_reshape_keeps_root_and_expanded_readers(device, operation):
    x = source(device)
    t = x.t()
    y = t.unsqueeze(0) if operation == 'unsqueeze' else getattr(t, operation)(1, 3, 2)
    expanded = y.expand(2, 3, 2)
    assert y._base is x and expanded._base is x
    assert tuple(y.stride()) == (3, 1, 3)
    assert x.data_ptr() == y.data_ptr() == expanded.data_ptr()
    physical(expanded)
    with torch.no_grad():
        x.fill_(7)
    np.testing.assert_array_equal(_array(physical(expanded)), np.full((2, 3, 2), 7.))
    gradient, = torch.autograd.grad(expanded.sum(), (x,))
    np.testing.assert_array_equal(_array(physical(gradient)), np.full((2, 3), 2.))


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('operation', ['unsqueeze', 'view', 'reshape'])
@pytest.mark.parametrize('reactivate_before_view', [False, True])
def test_fast_view_consumes_actual_history_at_operation_entry(
        device, operation, reactivate_before_view):
    x = source(device)
    y = x.view(2, 3)
    x.requires_grad_(False)
    with torch.no_grad():
        x.fill_(7)
    if reactivate_before_view:
        x.requires_grad_(True)
    z = y.unsqueeze(0) if operation == 'unsqueeze' else getattr(y, operation)(2, 3)
    if not reactivate_before_view:
        x.requires_grad_(True)
    weights = torch.arange(1, 7, device=device, dtype=torch.float32).reshape(z.shape)
    gx, gy, gz = torch.autograd.grad((z * weights).sum(), (x, y, z), allow_unused=True)
    if reactivate_before_view:
        np.testing.assert_array_equal(_array(physical(gx)), _array(weights).reshape(2, 3))
    else:
        assert gx is None
    np.testing.assert_array_equal(_array(physical(gy)), _array(weights).reshape(2, 3))
    np.testing.assert_array_equal(_array(physical(gz)), _array(weights))
