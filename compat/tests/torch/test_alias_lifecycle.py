"""Real Torch view creation, flag transitions and in-place history contracts."""

import gc
import warnings
from contextlib import nullcontext

import numpy as np
import pytest
import torch

from _rng_init_support import _array as _host_array, preserve_random_streams, require_rng_device


@pytest.fixture(autouse=True)
def forbid_fallbacks():
    with preserve_random_streams():
        yield


def source(device, initial=True):
    require_rng_device(device)
    value = torch.tensor([1., 2.], device=device, requires_grad=initial)
    if hasattr(torch, '_torch_compat_install_context'):
        value.sync()
        assert value.location() == ('device' if device == 'cuda' else 'cpu')
    return value


def _array(value):
    if hasattr(torch, '_torch_compat_install_context'):
        value.sync()
        # Empty values have no allocation to inspect. Their tests check device
        # and shape separately, without counting them as physical GPU math.
        if value.numel():
            assert value.location() == ('device' if value.device.type == 'cuda' else 'cpu')
    return _host_array(value)


def _physical_views():
    if hasattr(torch, '_torch_compat_install_context'):
        import jittor as jt
        return jt.flag_scope(transpose_storage_view=1)
    return nullcontext()


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('initial', [False, True])
@pytest.mark.parametrize('creation_no_grad', [False, True])
@pytest.mark.parametrize('change', ['source_false', 'source_true', 'alias_false', 'alias_true'])
def test_alias_flag_transition(device, initial, creation_no_grad, change):
    x = source(device, initial)
    with torch.set_grad_enabled(not creation_no_grad):
        y = torch.Tensor(x)
    target = x if change.startswith('source') else y
    if initial and not creation_no_grad and change == 'alias_false':
        with pytest.raises(RuntimeError, match='leaf variables'):
            target.requires_grad_(False)
        assert y.requires_grad and not y.is_leaf and y.grad_fn is not None
        return
    target.requires_grad_(change.endswith('true'))
    x_enabled = change.endswith('true') if change.startswith('source') else initial
    inherited = creation_no_grad or not initial
    y_enabled = (change == 'alias_true') or (x_enabled if inherited else initial)
    assert x.requires_grad == x_enabled
    assert y.requires_grad == y_enabled
    assert y.is_leaf == inherited
    assert (y.grad_fn is None) == inherited
    loss = (y * 3).sum()
    assert loss.requires_grad == y_enabled
    if y_enabled:
        targets = (x, y) if x_enabled else (y,)
        gradients = torch.autograd.grad(loss, targets, allow_unused=True)
        edge = not inherited or change == 'alias_true'
        for tensor, gradient in zip(targets, gradients):
            expected = edge and (tensor is y or not inherited)
            if expected:
                np.testing.assert_array_equal(_array(gradient), [3., 3.])
            else:
                assert gradient is None


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('mutation_grad', [False, True])
@pytest.mark.parametrize('inspection_grad', [False, True])
@pytest.mark.parametrize('target_name', ['source', 'alias'])
@pytest.mark.parametrize('method', ['fill', 'self_copy'])
def test_no_grad_alias_mutation_history(device, mutation_grad, inspection_grad, target_name, method):
    x = source(device)
    with torch.no_grad():
        y = torch.Tensor(x)
    target = x if target_name == 'source' else y
    def write():
        with torch.set_grad_enabled(mutation_grad):
            target.fill_(7.) if method == 'fill' else target.copy_(target)
    if mutation_grad:
        message = 'no_grad mode' if target_name == 'alias' else 'leaf Variable'
        with pytest.raises(RuntimeError, match=message):
            write()
        np.testing.assert_array_equal(_array(x), [1., 2.])
    else:
        write()
        np.testing.assert_array_equal(_array(x), [7., 7.] if method == 'fill' else [1., 2.])
    with torch.set_grad_enabled(inspection_grad):
        if mutation_grad:
            assert y.is_leaf and y.grad_fn is None
            assert (y * 3).sum().requires_grad == inspection_grad
        else:
            for inspect in (lambda: y.is_leaf, lambda: y.grad_fn):
                with pytest.raises(RuntimeError, match='no_grad mode'):
                    inspect()
            if inspection_grad:
                with pytest.raises(RuntimeError, match='no_grad mode'):
                    y * 3
            else:
                value = y * 3
                assert not value.requires_grad
                np.testing.assert_array_equal(_array(value),
                                              [21., 21.] if method == 'fill' else [3., 6.])


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('initial', [False, True])
@pytest.mark.parametrize('creation_no_grad', [False, True])
def test_inherited_flag_survives_source_owner(device, initial, creation_no_grad):
    x = source(device, initial)
    with torch.set_grad_enabled(not creation_no_grad):
        y = torch.Tensor(x)
    x.requires_grad_(True)
    del x
    gc.collect()
    assert y.requires_grad
    recorded = initial and not creation_no_grad
    assert y.is_leaf == (not recorded)
    gradient, = torch.autograd.grad((y * 3).sum(), (y,), allow_unused=True)
    if recorded:
        np.testing.assert_array_equal(_array(gradient), [3., 3.])
    else:
        assert gradient is None


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_disabling_activated_unrecorded_alias_restores_inheritance(device):
    x = source(device)
    with torch.no_grad():
        y = torch.Tensor(x)
    y.requires_grad_(True)
    y.requires_grad_(False)
    assert y.requires_grad and y.is_leaf and y.grad_fn is None
    gradient, = torch.autograd.grad((y * 3).sum(), (y,), allow_unused=True)
    assert gradient is None
    x.requires_grad_(False)
    assert not y.requires_grad


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('initial', [False, True])
@pytest.mark.parametrize('creation_no_grad', [False, True])
@pytest.mark.parametrize('target_name', ['source', 'alias'])
@pytest.mark.parametrize('method', ['fill', 'copy', 'self_copy'])
@pytest.mark.parametrize('after', ['keep', 'enable_source', 'disable_source', 'enable_alias'])
def test_written_alias_flag_transitions(device, initial, creation_no_grad, target_name, method, after):
    x = source(device, initial)
    with torch.set_grad_enabled(not creation_no_grad):
        y = torch.Tensor(x)
    target = x if target_name == 'source' else y
    with torch.no_grad():
        if method == 'fill':
            target.fill_(7.)
        else:
            target.copy_(target if method == 'self_copy' else torch.tensor([7., 8.], device=device))
    if after == 'enable_source':
        x.requires_grad_(True)
    elif after == 'disable_source':
        x.requires_grad_(False)
    elif after == 'enable_alias':
        y.requires_grad_(True)
    x_enabled = True if after == 'enable_source' else False if after == 'disable_source' else initial
    y_enabled = (after == 'enable_alias') or (x_enabled if creation_no_grad or not initial else True)
    assert x.requires_grad == x_enabled
    assert y.requires_grad == y_enabled
    np.testing.assert_array_equal(_array(x), [7., 7.] if method == 'fill' else [7., 8.] if method == 'copy' else [1., 2.])
    if creation_no_grad and x_enabled:
        for inspect in (lambda: y.is_leaf, lambda: y.grad_fn, lambda: y * 3):
            with pytest.raises(RuntimeError, match='no_grad mode'):
                inspect()
        return
    recorded = not creation_no_grad and (initial or x_enabled)
    assert y.is_leaf == (not recorded)
    assert (y.grad_fn is not None) == recorded
    loss = (y * 3).sum()
    assert loss.requires_grad == y_enabled
    if y_enabled:
        targets = (x, y) if x_enabled else (y,)
        for gradient in torch.autograd.grad(loss, targets, allow_unused=True):
            np.testing.assert_array_equal(_array(gradient), [3., 3.])


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_mutation_history_survives_source_owner(device):
    x = source(device)
    with torch.no_grad():
        y = torch.Tensor(x)
        x.copy_(x)
    del x
    gc.collect()
    assert y.requires_grad
    for inspect in (lambda: y.is_leaf, lambda: y.grad_fn, lambda: y * 3):
        with pytest.raises(RuntimeError, match='no_grad mode'):
            inspect()


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('restore', [False, True])
@pytest.mark.parametrize('custom', [False, True])
def test_recorded_alias_edges_survive_later_inherited_flag(device, restore, custom):
    class Product(torch.autograd.Function):
        @staticmethod
        def forward(ctx, left, right):
            ctx.save_for_backward(left, right)
            return left * right * 5

        @staticmethod
        def backward(ctx, gradient):
            left, right = ctx.saved_tensors
            return gradient * right * 5, gradient * left * 5

    x = source(device)
    with torch.no_grad():
        y = torch.Tensor(x)
    y.requires_grad_(True)
    old = (y * 3).sum()
    y.requires_grad_(False)
    if restore:
        y.requires_grad_(True)
    assert y.requires_grad
    new = (Product.apply(y, x) if custom else y * x * 5).sum()
    for loss, expected_x, expected_y in (
        (old, None, [3., 3.]),
        (new, [5., 10.], [5., 10.] if restore else None),
        (old + new, [5., 10.], [8., 13.] if restore else [3., 3.]),
    ):
        gradients = torch.autograd.grad(loss, (x, y), allow_unused=True, retain_graph=True)
        for gradient, expected in zip(gradients, (expected_x, expected_y)):
            if expected is None:
                assert gradient is None
            else:
                np.testing.assert_array_equal(_array(gradient), expected)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('creation_no_grad', [False, True])
@pytest.mark.parametrize('method', ['fill', 'copy', 'self_copy'])
def test_source_frozen_at_write_does_not_reconnect_recorded_view(device, creation_no_grad, method):
    x = source(device)
    with torch.set_grad_enabled(not creation_no_grad):
        y = torch.Tensor(x)
    x.requires_grad_(False)
    with torch.no_grad():
        if method == 'fill':
            x.fill_(7.)
        else:
            x.copy_(x if method == 'self_copy' else torch.tensor([7., 8.], device=device))
    if creation_no_grad:
        assert not y.requires_grad and y.is_leaf and y.grad_fn is None
        with pytest.raises(RuntimeError, match='does not require grad'):
            torch.autograd.grad((y * 3).sum(), (y,), allow_unused=True)
    else:
        assert y.requires_grad and not y.is_leaf and y.grad_fn is not None
        gradient, = torch.autograd.grad((y * 3).sum(), (y,), retain_graph=True)
        np.testing.assert_array_equal(_array(gradient), [3., 3.])
    x.requires_grad_(True)
    if creation_no_grad:
        for inspect in (lambda: y.is_leaf, lambda: y.grad_fn, lambda: y * 3):
            with pytest.raises(RuntimeError, match='no_grad mode'):
                inspect()
    else:
        assert y.requires_grad and not y.is_leaf and y.grad_fn is not None
        source_gradient, alias_gradient = torch.autograd.grad(
            (y * 3).sum(), (x, y), allow_unused=True)
        assert source_gradient is None
        np.testing.assert_array_equal(_array(alias_gradient), [3., 3.])


@pytest.mark.parametrize('device', ['cuda'])
def test_custom_backward_keeps_explicit_cross_device_input_contract(device):
    class TransferProduct(torch.autograd.Function):
        @staticmethod
        def forward(ctx, left, right):
            ctx.save_for_backward(left, right)
            return left * right.to(left.device)

        @staticmethod
        def backward(ctx, gradient):
            left, right = ctx.saved_tensors
            return gradient * right.to(left.device), (gradient * left).to(right.device)

    left = source(device)
    right = torch.tensor([3., 4.], device='cpu', requires_grad=True)
    gradients = torch.autograd.grad(TransferProduct.apply(left, right).sum(), (left, right))
    assert gradients[0].device.type == 'cuda' and gradients[1].device.type == 'cpu'
    np.testing.assert_array_equal(_array(gradients[0]), [3., 4.])
    np.testing.assert_array_equal(_array(gradients[1]), [1., 2.])


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('creation_no_grad', [False, True])
def test_integer_alias_does_not_attempt_gradient_activation(device, creation_no_grad):
    require_rng_device(device)
    x = torch.tensor([1, 2], dtype=torch.int64, device=device)
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter('always')
        with torch.set_grad_enabled(not creation_no_grad):
            y = torch.Tensor(x)
    assert not records
    assert y.dtype == torch.int64 and not y.requires_grad
    assert y.is_leaf and y.grad_fn is None
    with torch.no_grad():
        x.copy_(x)
    assert y.is_leaf and y.grad_fn is None
    np.testing.assert_array_equal(_array(y), [1., 2.])


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('size', [0, 2])
@pytest.mark.parametrize('writer', ['source', 'alias', 'detached'])
def test_self_copy_updates_shared_version_even_without_elements(device, size, writer):
    require_rng_device(device)
    x = torch.tensor([1., 2.] if size else [], device=device, requires_grad=True)
    with torch.no_grad():
        y = torch.Tensor(x)
    target = x if writer == 'source' else y if writer == 'alias' else x.detach()
    with torch.no_grad():
        target.copy_(target)
    assert y.requires_grad
    for inspect in (lambda: y.is_leaf, lambda: y.grad_fn, lambda: y * 3):
        with pytest.raises(RuntimeError, match='no_grad mode'):
            inspect()
    with torch.no_grad():
        assert not (y * 3).requires_grad


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_detach_shares_version_but_has_independent_trainability(device):
    x = source(device)
    with torch.no_grad():
        y = torch.Tensor(x)
    detached = x.detach()
    assert detached.is_leaf and not detached.requires_grad and detached.grad_fn is None
    x.requires_grad_(False)
    x.requires_grad_(True)
    assert not detached.requires_grad
    assert y.requires_grad and y.is_leaf and y.grad_fn is None
    detached.requires_grad_(True)
    gradients = torch.autograd.grad((detached * 3).sum(), (x, y, detached), allow_unused=True)
    assert gradients[0] is None and gradients[1] is None
    np.testing.assert_array_equal(_array(gradients[2]), [3., 3.])
    detached.requires_grad_(False)
    assert x.requires_grad and y.requires_grad and not detached.requires_grad
    assert y.is_leaf and y.grad_fn is None


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('size', [0, 2])
@pytest.mark.parametrize('method', ['fill', 'self_copy'])
def test_detached_write_preserves_lazy_recorded_view_source_edge(device, size, method):
    require_rng_device(device)
    x = torch.tensor([1., 2.] if size else [], device=device, requires_grad=True)
    y = torch.Tensor(x)
    x.requires_grad_(False)
    detached = x.detach()
    with torch.no_grad():
        detached.fill_(7.) if method == 'fill' else detached.copy_(detached)
    x.requires_grad_(True)
    assert y.requires_grad and not y.is_leaf and y.grad_fn is not None
    gradients = torch.autograd.grad((y * 3).sum(), (x, y), allow_unused=True)
    assert gradients[0] is not None
    np.testing.assert_array_equal(_array(gradients[0]), [3., 3.] if size else [])
    assert gradients[1] is not None and gradients[1].shape == x.shape
    np.testing.assert_array_equal(_array(gradients[1]), [3., 3.] if size else [])


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('size', [0, 2])
@pytest.mark.parametrize('writer', ['source', 'alias', 'detached'])
@pytest.mark.parametrize('method', ['fill', 'self_copy'])
@pytest.mark.parametrize('inspect_before_enable', [False, True])
@pytest.mark.parametrize('initial', [False, True])
def test_recorded_view_consumes_write_epoch_at_history_observation(
        device, size, writer, method, inspect_before_enable, initial):
    require_rng_device(device)
    x = torch.tensor([1., 2.] if size else [], device=device, requires_grad=initial)
    y = torch.Tensor(x)
    x.requires_grad_(False)
    target = x if writer == 'source' else y if writer == 'alias' else x.detach()
    with torch.no_grad():
        target.fill_(7.) if method == 'fill' else target.copy_(target)
    assert y.requires_grad == initial
    if inspect_before_enable:
        assert (y.grad_fn is not None) == initial
    x.requires_grad_(True)
    gradients = torch.autograd.grad((y * 3).sum(), (x, y), allow_unused=True)
    expected_gradient = [3., 3.] if size else []
    if initial and inspect_before_enable:
        assert gradients[0] is None
    else:
        assert gradients[0] is not None
        np.testing.assert_array_equal(_array(gradients[0]), expected_gradient)
    assert gradients[1] is not None
    np.testing.assert_array_equal(_array(gradients[1]), expected_gradient)
    expected_value = ([7., 7.] if method == 'fill' else [1., 2.]) if size else []
    for value in (x, y, target):
        np.testing.assert_array_equal(_array(value), expected_value)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_metadata_and_no_grad_reads_do_not_consume_view_history(device):
    x = source(device)
    y = torch.Tensor(x)
    x.requires_grad_(False)
    with torch.no_grad():
        x.copy_(x)
        observed = y * 2
    assert y.requires_grad and not x.requires_grad
    assert tuple(y.shape) == (2,) and y.device.type == device
    assert y.data_ptr() == x.data_ptr()
    if hasattr(torch, '_torch_compat_install_context'):
        import jittor as jt
        jt.core.dispatch_context([y])
    x.requires_grad_(True)
    gradients = torch.autograd.grad((y * 3).sum(), (x, y))
    for gradient in gradients:
        np.testing.assert_array_equal(_array(gradient), [3., 3.])
    np.testing.assert_array_equal(_array(observed), [2., 4.])


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('action', ['source_true', 'alias_true', 'alias_false'])
def test_pending_view_flag_validation_does_not_consume_history(device, action):
    x = source(device)
    y = torch.Tensor(x)
    x.requires_grad_(False)
    with torch.no_grad():
        x.copy_(x)
    if action == 'alias_false':
        with pytest.raises(RuntimeError, match='leaf variables'):
            y.requires_grad_(False)
    else:
        (x if action == 'source_true' else y).requires_grad_(True)
    x.requires_grad_(True)
    for gradient in torch.autograd.grad((y * 3).sum(), (x, y)):
        np.testing.assert_array_equal(_array(gradient), [3., 3.])


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('kind', ['data', 'detach'])
@pytest.mark.parametrize('mutation_grad', [False, True])
@pytest.mark.parametrize('method', ['fill', 'copy', 'self_copy', 'add', 'sub'])
def test_data_and_detach_have_distinct_shared_storage_versions(
        device, kind, mutation_grad, method):
    x = source(device)
    with torch.no_grad():
        y = torch.Tensor(x)
    writer = x.data if kind == 'data' else x.detach()
    with torch.set_grad_enabled(mutation_grad):
        if method == 'fill':
            writer.fill_(7.)
        elif method == 'copy':
            writer.copy_(torch.tensor([7., 8.], device=device))
        elif method == 'self_copy':
            writer.copy_(writer)
        elif method == 'add':
            writer.add_(2.)
        else:
            writer.sub_(2.)
    assert not writer.requires_grad and writer.is_leaf and writer.grad_fn is None
    expected = {'fill': [7., 7.], 'copy': [7., 8.], 'self_copy': [1., 2.],
                'add': [3., 4.], 'sub': [-1., 0.]}[method]
    for value in (x, writer):
        np.testing.assert_array_equal(_array(value), expected)
    assert x.data_ptr() == writer.data_ptr()
    if kind == 'detach':
        for inspect in (lambda: y.is_leaf, lambda: y.grad_fn, lambda: y * 3):
            with pytest.raises(RuntimeError, match='no_grad mode'):
                inspect()
    else:
        assert y.is_leaf and y.grad_fn is None
        assert torch.autograd.grad((y * 3).sum(), (x, y), allow_unused=True) == (None, None)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('initial', [False, True])
@pytest.mark.parametrize('replacement_grad', [False, True])
def test_data_replacement_keeps_old_view_storage_and_independent_metadata(
        device, initial, replacement_grad):
    x = source(device, initial)
    replacement = torch.tensor([3., 4.], device=device, requires_grad=replacement_grad)
    with torch.no_grad():
        y = torch.Tensor(x)
    x.data = replacement
    assert x.requires_grad == initial and y.requires_grad == initial
    assert replacement.requires_grad == replacement_grad
    assert x.data_ptr() == replacement.data_ptr() and y.data_ptr() != x.data_ptr()
    np.testing.assert_array_equal(_array(x), [3., 4.])
    np.testing.assert_array_equal(_array(y), [1., 2.])
    assert y.is_leaf and y.grad_fn is None
    if initial:
        targets = (x, replacement) if replacement_grad else (x,)
        gradients = torch.autograd.grad((x * 3).sum(), targets, allow_unused=True)
        np.testing.assert_array_equal(_array(gradients[0]), [3., 3.])
        if replacement_grad:
            assert gradients[1] is None
        assert torch.autograd.grad((y * 3).sum(), (x, y), allow_unused=True) == (None, None)
    replacement.requires_grad_(False)
    assert x.requires_grad == initial and y.requires_grad == initial


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('kind', ['data', 'detach'])
@pytest.mark.parametrize('creation_grad', [False, True])
def test_detaching_a_stale_view_does_not_observe_its_history(device, kind, creation_grad):
    x = source(device)
    with torch.no_grad():
        y = torch.Tensor(x)
        x.copy_(x)
    with torch.set_grad_enabled(creation_grad):
        detached = y.data if kind == 'data' else y.detach()
    assert not detached.requires_grad and detached.is_leaf and detached.grad_fn is None
    with pytest.raises(RuntimeError, match='no_grad mode'):
        y.clone()
    with pytest.raises(RuntimeError, match='no_grad mode'):
        _ = y.grad_fn
    detached.requires_grad_(True)
    gradients = torch.autograd.grad((detached * 3).sum(), (x, detached), allow_unused=True)
    assert gradients[0] is None
    np.testing.assert_array_equal(_array(gradients[1]), [3., 3.])


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_scalar_data_alias_preserves_shape_and_source_gradient(device):
    require_rng_device(device)
    x = torch.tensor(2., device=device, requires_grad=True)
    data = x.data
    assert data.shape == x.shape == () and not data.requires_grad
    data.add_(2.)
    assert data.data_ptr() == x.data_ptr()
    assert x.item() == 4. and data.item() == 4.
    gradient, = torch.autograd.grad(x * x, (x,))
    assert gradient.shape == x.shape
    np.testing.assert_array_equal(_array(gradient), 8.)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('kind', ['leaf_old', 'leaf_square', 'nonleaf_old', 'nonleaf_new',
                                  'view_old', 'view_new', 'constructor_view'])
@pytest.mark.parametrize('replacement', ['fresh', 'self', 'detached'])
def test_data_storage_replacement_retains_recorded_gradient_history(device, kind, replacement):
    x = source(device)
    if kind.startswith('nonleaf'):
        y = x * 2
    elif kind.startswith('view'):
        y = x[:]
    elif kind == 'constructor_view':
        y = torch.Tensor(x)
    else:
        y = x
    old_loss = ((y ** 2) if kind == 'leaf_square' else (y * 3)).sum()
    recorded = y.grad_fn is not None
    rhs = (torch.tensor([4., 5.], device=device) if replacement == 'fresh'
           else y if replacement == 'self' else y.detach())
    y.data = rhs
    assert (y.grad_fn is not None) == recorded
    assert y.data_ptr() == rhs.data_ptr()
    loss = (y * 3).sum() if kind.endswith('_new') else old_loss
    loss.backward()
    if kind == 'leaf_square':
        gradient = [8., 10.] if replacement == 'fresh' else [2., 4.]
    else:
        gradient = [6., 6.] if kind.startswith('nonleaf') else [3., 3.]
    np.testing.assert_array_equal(_array(x.grad), gradient)
    assert old_loss.item() == (5. if kind == 'leaf_square'
                               else 18. if kind.startswith('nonleaf') else 9.)
    if kind.endswith('_new'):
        assert loss.item() == (27. if replacement == 'fresh'
                               else 18. if kind.startswith('nonleaf') else 9.)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('scalar_source', [False, True])
@pytest.mark.parametrize('old_graph', [False, True])
def test_data_shape_replacement_keeps_recorded_gradient_shape(device, scalar_source, old_graph):
    require_rng_device(device)
    x = torch.tensor(2. if scalar_source else [1., 2.], device=device, requires_grad=True)
    old_loss = (x * 3).sum()
    x.data = torch.tensor([4., 5.] if scalar_source else 4., device=device)
    assert x.shape == ((2,) if scalar_source else ())
    loss = old_loss if old_graph else (x * 3).sum()
    if not scalar_source and not old_graph:
        with pytest.raises(RuntimeError, match='invalid gradient'):
            loss.backward()
        return
    loss.backward()
    gradient = 3. if scalar_source and old_graph else 6. if scalar_source else [3., 3.]
    np.testing.assert_array_equal(_array(x.grad), gradient)
    assert old_loss.item() == (6. if scalar_source else 9.)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('initial', [False, True])
def test_data_shape_without_recorded_history_uses_new_shape(device, initial):
    require_rng_device(device)
    x = torch.tensor(2., device=device, requires_grad=initial)
    x.data = torch.tensor([4., 5.], device=device)
    x.requires_grad_(True)
    (x * 3).sum().backward()
    np.testing.assert_array_equal(_array(x.grad), [3., 3.])


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('layout', ['contiguous', 'transpose', 'slice', 'expand'])
def test_leaf_gradient_layout_preserves_values_without_aliasing_input(device, layout):
    require_rng_device(device)
    base = torch.tensor([[1., 2.], [3., 4.]], device=device)
    with _physical_views():
        value = (base.t() if layout == 'transpose' else base[:, :1] if layout == 'slice'
                 else base[:1, :].expand(2, 2) if layout == 'expand' else base)
    x = value.detach().requires_grad_(True)
    expected = np.arange(1, x.numel() + 1, dtype=np.float32).reshape(tuple(x.shape))
    coefficients = torch.tensor(expected, device=device, dtype=x.dtype)
    loss = (x * coefficients).sum()
    returned, = torch.autograd.grad(loss, (x,), retain_graph=True)
    np.testing.assert_array_equal(_array(returned), expected)
    loss.backward()
    np.testing.assert_array_equal(_array(x.grad), expected)
    assert x.grad.data_ptr() != x.data_ptr()
    assert x.grad.stride() == ((1, 2) if layout == 'transpose' else
                               (1, 1) if layout == 'slice' else (2, 1))
    before = _array(x).copy()
    x.grad.fill_(0.)
    np.testing.assert_array_equal(_array(x), before)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_leaf_gradient_strided_allocation_preserves_second_order_graph(device):
    require_rng_device(device)
    with _physical_views():
        x = torch.tensor([[1., 2.], [3., 4.]], device=device).t().detach().requires_grad_(True)
    (x ** 3).sum().backward(create_graph=True)
    try:
        assert x.grad.stride() == (1, 2) and x.grad.requires_grad
        np.testing.assert_array_equal(_array(x.grad), [[3., 27.], [12., 48.]])
        second, = torch.autograd.grad(x.grad.sum(), (x,))
        np.testing.assert_array_equal(_array(second), [[6., 18.], [12., 24.]])
    finally:
        x.grad = None


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('old_shape,new_shape', [((), (2,)), ((2,), ()), ((1,), (2,)),
                                               ((2,), (1,)), ((2,), (2, 2)), ((2, 2), (2,))])
@pytest.mark.parametrize('old_graph', [False, True])
def test_data_shape_graph_gradient_and_leaf_layout_are_separate(device, old_shape, new_shape, old_graph):
    require_rng_device(device)
    x = torch.ones(old_shape, device=device, requires_grad=True)
    old_loss = (x * 3).sum()
    x.data = torch.ones(new_shape, device=device)
    loss = old_loss if old_graph else (x * 3).sum()
    compatible = (len(new_shape) >= len(old_shape)
                  and all(a == b or a == 1 for a, b in zip(reversed(old_shape), reversed(new_shape))))
    if not old_graph and not compatible:
        with pytest.raises(RuntimeError, match='invalid gradient'):
            torch.autograd.grad(loss, (x,), retain_graph=True)
        return
    gradient, = torch.autograd.grad(loss, (x,), retain_graph=True)
    factor = (np.prod(new_shape) if new_shape else 1) / (np.prod(old_shape) if old_shape else 1)
    np.testing.assert_array_equal(_array(gradient), np.full(old_shape, 3. if old_graph else 3. * factor))
    if old_shape == (2, 2) and new_shape == (2,):
        with pytest.raises(RuntimeError, match='broadcast'):
            loss.backward()
    else:
        loss.backward()
        expected_shape = new_shape if new_shape == (2, 2) else old_shape
        np.testing.assert_array_equal(_array(x.grad), np.full(expected_shape, 3. if old_graph else 3. * factor))


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('replace_owner', ['source', 'alias'])
@pytest.mark.parametrize('constructor', [False, True])
@pytest.mark.parametrize('nested', [False, True])
def test_data_orphaned_views_keep_leaf_write_guards(device, replace_owner, constructor, nested):
    x = source(device)
    y = torch.Tensor(x) if constructor else x[:]
    destination = x if replace_owner == 'source' else y
    destination.data = torch.tensor([4., 5.], device=device)
    target = y[:] if nested else y
    before = _array(target).copy()
    with pytest.raises(RuntimeError, match='view of a leaf Variable'):
        target.add_(1.)
    np.testing.assert_array_equal(_array(target), before)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('detached_leaf', [False, True])
def test_data_orphan_guard_does_not_look_through_ordinary_clone(device, detached_leaf):
    x = source(device)
    base = x.detach().requires_grad_(True) if detached_leaf else x.clone()
    alias = base[:]
    base.data = torch.tensor([4., 5.], device=device)
    if detached_leaf:
        with pytest.raises(RuntimeError, match='view of a leaf Variable'):
            alias.add_(1.)
    else:
        alias.add_(1.)
        alias.sum().backward()
        np.testing.assert_array_equal(_array(alias), [2., 3.])
        np.testing.assert_array_equal(_array(x.grad), [1., 1.])


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('scenario', ['leaf_mutate_input', 'nonleaf_mutate_input',
                                     'constructor_mutate_input', 'explicit_copy_mutate_source',
                                     'explicit_view_mutate_source', 'nonleaf_mutate_source'])
@pytest.mark.parametrize('operation', ['multiply', 'sin', 'cos', 'log', 'sqrt', 'exp', 'tanh'])
def test_layout_forward_preserves_logical_saved_inputs_and_outputs(device, scenario, operation):
    require_rng_device(device)
    with _physical_views():
        leaf = torch.tensor([[.2, .4], [.6, .8]], device=device).t().detach().requires_grad_()
        if scenario.startswith('nonleaf'):
            value = leaf * 2
        elif scenario.startswith('constructor'):
            value = torch.Tensor(leaf)
        elif scenario.startswith('explicit_copy'):
            value = leaf.contiguous()
        elif scenario.startswith('explicit_view'):
            value = leaf.t()
        else:
            value = leaf
    original = _array(value).copy()
    coefficient_values = np.array([[1., 2.], [3., 4.]], dtype=np.float32)
    coefficients = torch.tensor(coefficient_values, device=device)
    output = value * value if operation == 'multiply' else getattr(value, operation)()
    saved_output = _array(output).copy()
    assert output.stride() == value.stride()
    loss = (output * coefficients).sum()
    mutation = leaf if scenario.endswith('_source') else value
    replacement = np.array([[1.1, 1.2], [1.3, 1.4]], dtype=np.float32)
    mutation.data = torch.tensor(replacement, device=device)
    # Explicit user views/copies keep their old storage after source .data is
    # replaced. Internal layout operands cannot replace the logical input.
    current = original if scenario.endswith('_source') else replacement
    np.testing.assert_array_equal(_array(value), current)
    np.testing.assert_array_equal(_array(output), saved_output)
    derivative = (2 * current if operation == 'multiply' else
                  np.cos(current) if operation == 'sin' else
                  -np.sin(current) if operation == 'cos' else
                  1 / current if operation == 'log' else
                  .5 / saved_output if operation == 'sqrt' else
                  saved_output if operation == 'exp' else
                  1 - saved_output * saved_output)
    expected = coefficient_values * derivative
    targets = (value,) if value is leaf else (value, leaf)
    gradients = torch.autograd.grad(loss, targets)
    np.testing.assert_allclose(_array(gradients[0]), expected, rtol=2e-6, atol=2e-7)
    if len(gradients) == 2:
        expected_leaf = (expected * 2 if scenario.startswith('nonleaf') else
                         expected.T if scenario.startswith('explicit_view') else expected)
        np.testing.assert_allclose(_array(gradients[1]), expected_leaf, rtol=2e-6, atol=2e-7)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('old_shape,new_shape', [((2, 2), (4,)), ((4,), (2, 2)),
                                               ((2, 1, 2), (4,)), ((4,), (2, 1, 2)),
                                               ((2, 3), (3, 2)), ((1, 2, 2), (2, 2))])
def test_data_equal_numel_rank_change_keeps_old_graph_gradient_but_refuses_leaf_copy(
        device, old_shape, new_shape):
    require_rng_device(device)
    x = torch.ones(old_shape, device=device, requires_grad=True)
    expected = np.arange(1, x.numel() + 1, dtype=np.float32).reshape(old_shape)
    loss = (x * torch.tensor(expected, device=device)).sum()
    x.data = torch.ones(new_shape, device=device)
    gradient, = torch.autograd.grad(loss, (x,), retain_graph=True)
    np.testing.assert_array_equal(_array(gradient), expected)
    assert tuple(gradient.shape) == old_shape
    with pytest.raises(RuntimeError, match='(?i)size|shape|broadcast'):
        loss.backward()


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('new_transpose', [False, True])
def test_data_layout_transition_uses_nonuniform_new_input_and_current_leaf_stride(device, new_transpose):
    require_rng_device(device)
    with _physical_views():
        original = torch.tensor([[1., 2.], [3., 4.]], device=device)
        x = (original if new_transpose else original.t()).detach().requires_grad_()
    coefficient_values = np.array([[1., 2.], [3., 4.]], dtype=np.float32)
    loss = (x * x * torch.tensor(coefficient_values, device=device)).sum()
    with _physical_views():
        replacement = torch.tensor([[11., 12.], [13., 14.]], device=device)
        if new_transpose:
            replacement = replacement.t()
    current = _array(replacement).copy()
    x.data = replacement
    returned, = torch.autograd.grad(loss, (x,), retain_graph=True)
    expected = 2 * current * coefficient_values
    np.testing.assert_array_equal(_array(returned), expected)
    loss.backward()
    np.testing.assert_array_equal(_array(x.grad), expected)
    assert x.grad.stride() == ((1, 2) if new_transpose else (2, 1))


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_no_grad_layout_forward_still_uses_real_optimized_output(device):
    require_rng_device(device)
    with _physical_views():
        x = torch.tensor([[1., 2.], [3., 4.]], device=device).t().detach().requires_grad_()
    with torch.no_grad():
        output = x * x
    assert output.stride() == (1, 2)
    assert not output.requires_grad and output.grad_fn is None
    np.testing.assert_array_equal(_array(output), [[1., 9.], [4., 16.]])
    x.data = torch.full((2, 2), 10., device=device)
    np.testing.assert_array_equal(_array(output), [[1., 9.], [4., 16.]])


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('kind', ['recorded_view', 'nonleaf'])
@pytest.mark.parametrize('operation', ['multiply', 'sin', 'exp', 'negative'])
def test_pending_layout_consumer_keeps_explicit_target_and_source_edges(device, kind, operation):
    require_rng_device(device)
    with _physical_views():
        base = torch.tensor([[.1, .2], [.3, .4]], device=device, requires_grad=True)
        value = base.t()
        if kind == 'nonleaf':
            value = value * 2
    coefficient_values = np.array([[1., 2.], [3., 4.]], dtype=np.float32)
    coefficients = torch.tensor(coefficient_values, device=device)
    output = (value * value if operation == 'multiply' else
              -value if operation == 'negative' else getattr(value, operation)())
    loss = (output * coefficients).sum()
    # Do not read/synchronize value before the consumer is built: the old
    # layout fast path bypassed this explicit target only while it was lazy.
    replacement = np.array([[.5, .6], [.7, .8]], dtype=np.float32)
    value.data = torch.tensor(replacement, device=device)
    original = np.array([[.1, .3], [.2, .4]], dtype=np.float32) * (2 if kind == 'nonleaf' else 1)
    old_output = (original * original if operation == 'multiply' else
                  -original if operation == 'negative' else
                  np.sin(original) if operation == 'sin' else np.exp(original))
    np.testing.assert_allclose(_array(output), old_output, rtol=2e-6, atol=2e-7)
    np.testing.assert_array_equal(_array(value), replacement)
    derivative = (2 * replacement if operation == 'multiply' else
                  -np.ones_like(replacement) if operation == 'negative' else
                  np.cos(replacement) if operation == 'sin' else old_output)
    expected = coefficient_values * derivative
    gradient, base_gradient = torch.autograd.grad(loss, (value, base))
    np.testing.assert_allclose(_array(gradient), expected, rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(_array(base_gradient), expected.T * (2 if kind == 'nonleaf' else 1),
                               rtol=2e-6, atol=2e-7)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_rejected_layout_operator_does_not_disable_following_autograd(device):
    require_rng_device(device)
    with _physical_views():
        x = torch.tensor([[1., 2.], [3., 4.]], device=device).t().detach().requires_grad_()
    with pytest.raises(RuntimeError):
        torch.bitwise_and(x, x)
    gradient, = torch.autograd.grad((x * x).sum(), (x,))
    np.testing.assert_array_equal(_array(gradient), [[2., 6.], [4., 8.]])
