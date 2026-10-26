"""View eligibility and mutation history belong to the native graph owner."""

import gc
import sys

import numpy as np
import pytest

import jittor as jt
from jittor.autograd.policy import EXPLICIT_REQUIRES_GRAD, NATIVE, policy_scope
from jittor._runtime.fallback import forbid_backend_fallbacks


@pytest.fixture(autouse=True)
def native_only():
    assert 'jittor.compat.torch' not in sys.modules
    assert 'torch' not in sys.modules
    with forbid_backend_fallbacks():
        yield
        jt.sync_all(True)


def alias_of(source):
    alias = jt.Var.clone(source)
    return alias._set_view_of(source, Ellipsis, True)


def assert_device(value):
    value.sync()
    assert value.location() == ('device' if jt.flags.use_cuda else 'cpu')


def test_inherited_visibility_is_not_a_native_gradient_edge():
    with policy_scope(EXPLICIT_REQUIRES_GRAD):
        source = jt.array(np.array([1., 2.], dtype='float32'))
        assert_device(source)
        source.requires_grad = False
        with jt.no_grad():
            alias = alias_of(source)
        assert not alias.requires_grad
        source.requires_grad = True
        assert alias.requires_grad and alias.is_backward_leaf
        assert not alias._has_gradient_edge()
        loss = (alias * 3).sum()
        assert loss.requires_grad
        assert jt.core.grad_optional(loss, [source, alias], True) == [None, None]
        alias.requires_grad = True
        gradient, = jt.core.grad_optional((alias * 5).sum(), [alias], True)
        assert_device(gradient)
        np.testing.assert_array_equal(gradient.numpy(), [5., 5.])


@pytest.mark.parametrize('target', ['source', 'alias'])
@pytest.mark.parametrize('self_copy', [False, True])
def test_native_mutation_invalidates_no_grad_alias_inspection(target, self_copy):
    with policy_scope(EXPLICIT_REQUIRES_GRAD):
        source = jt.array(np.array([1., 2.], dtype='float32'))
        assert_device(source)
        with jt.no_grad():
            alias = alias_of(source)
        destination = source if target == 'source' else alias
        with pytest.raises(RuntimeError, match='no_grad mode|leaf Variable'):
            destination._check_inplace_autograd()
        with jt.no_grad():
            if self_copy:
                destination._copy_into(destination)
            else:
                destination.assign(jt.array(np.array([7., 7.], dtype='float32')))
        assert alias.requires_grad
        for inspect in (lambda: alias.is_backward_leaf,
                        lambda: alias.grad_fn_node_id):
            with pytest.raises(RuntimeError, match='no_grad mode'):
                inspect()
        with pytest.raises(RuntimeError, match='no_grad mode'):
            alias * 3
        with jt.no_grad():
            result = alias * 3
            assert not result.requires_grad
            np.testing.assert_array_equal(result.numpy(), [3., 6.] if self_copy else [21., 21.])


def test_policy_objects_do_not_pin_source_vars():
    with policy_scope(EXPLICIT_REQUIRES_GRAD):
        source = jt.array(np.array([1., 2.], dtype='float32'))
        source.requires_grad = False
        with jt.no_grad():
            alias = alias_of(source)
        source.requires_grad = True
        del source
        gc.collect()
        assert alias.requires_grad and not alias._has_gradient_edge()
        assert alias.is_backward_leaf and alias.grad_fn_node_id == -1


def test_nonleaf_flag_guard_is_selected_by_the_public_policy():
    with policy_scope(EXPLICIT_REQUIRES_GRAD):
        source = jt.array(np.array([1., 2.], dtype='float32'))
        result = source * 2
        result.requires_grad = True
        with pytest.raises(RuntimeError, match='leaf variables'):
            result.requires_grad = False
    with policy_scope(NATIVE):
        identity = result.grad_fn_node_id
        result.requires_grad = False
        assert result.is_backward_leaf
        result.requires_grad = True
        assert result.grad_fn_node_id == identity


def test_view_history_does_not_override_native_permanent_stop():
    with policy_scope(EXPLICIT_REQUIRES_GRAD):
        source = jt.array(np.array([1., 2.], dtype='float32'))
        alias = alias_of(source)
        assert not alias.is_backward_leaf
        source.stop_grad()
        assert alias.is_backward_leaf
        assert alias.grad_fn_node_id == -1


def test_rejected_tape_history_does_not_leave_partial_graph_edges():
    with policy_scope(EXPLICIT_REQUIRES_GRAD):
        source = jt.array(np.array([1., 2.], dtype='float32'))
        with jt.no_grad():
            alias = alias_of(source)
        alias.requires_grad = True
        taped_input = alias.tape()
        taped_output = (taped_input * 3).tape()
        with jt.no_grad():
            source._copy_into(source)
        with pytest.raises(RuntimeError, match='no_grad mode'):
            jt.tape_together([taped_input], [taped_output], lambda grad: (grad * 3,))
        del taped_output, taped_input, alias
        gc.collect()
        gradient, = jt.core.grad_optional((source * 5).sum(), [source], True)
        assert_device(gradient)
        np.testing.assert_array_equal(gradient.numpy(), [5., 5.])


def test_permanent_stop_ends_already_observed_rebased_view_history():
    with policy_scope(EXPLICIT_REQUIRES_GRAD):
        source = jt.array(np.array([1., 2.], dtype='float32'))
        assert_device(source)
        alias = alias_of(source)
        source.requires_grad = False
        with jt.no_grad():
            source._copy_into(source)
        assert alias.grad_fn_node_id >= 0
        source.stop_grad()
        assert alias.is_backward_leaf
        assert alias.grad_fn_node_id == -1


@pytest.mark.parametrize('observe_before_enable', [False, True])
def test_native_history_query_consumes_epoch_but_metadata_does_not(observe_before_enable):
    with policy_scope(EXPLICIT_REQUIRES_GRAD):
        source = jt.array(np.array([1., 2.], dtype='float32'))
        alias = alias_of(source)
        source.requires_grad = False
        with jt.no_grad():
            source._copy_into(source)
        jt.core.dispatch_context([alias])
        assert alias.requires_grad and not source.requires_grad
        assert not alias._has_leaf_accumulator()
        if observe_before_enable:
            assert alias.grad_fn_node_id >= 0
        source.requires_grad = True
        gradients = jt.core.grad_optional((alias * 3).sum(), [source, alias], True)
        if observe_before_enable:
            assert gradients[0] is None
        else:
            assert_device(gradients[0])
            np.testing.assert_array_equal(gradients[0].numpy(), [3., 3.])
        assert_device(gradients[1])
        np.testing.assert_array_equal(gradients[1].numpy(), [3., 3.])


def test_native_data_alias_and_detach_have_independent_version_policies():
    with policy_scope(EXPLICIT_REQUIRES_GRAD):
        source = jt.array(np.array([1., 2.], dtype='float32'))
        assert_device(source)
        with jt.no_grad():
            alias = alias_of(source)
        data = source._storage_data_alias()
        assert not data.requires_grad
        data._copy_into(data)
        assert alias.requires_grad and alias.is_backward_leaf
        assert alias.grad_fn_node_id == -1
        detached = source.detach().stop_grad()
        detached._copy_into(detached)
        with pytest.raises(RuntimeError, match='no_grad mode'):
            _ = alias.is_backward_leaf


@pytest.mark.parametrize('nonleaf', [False, True])
@pytest.mark.parametrize('square', [False, True])
def test_native_storage_replacement_keeps_var_and_recorded_producer(nonleaf, square):
    with policy_scope(EXPLICIT_REQUIRES_GRAD):
        x = jt.array(np.array([1., 2.], dtype='float32'))
        y = x * 2 if nonleaf else x
        old_loss = (y ** 2).sum() if square else (y * 3).sum()
        producer = y.grad_fn_node_id
        rhs = jt.array(np.array([4., 5.], dtype='float32'))
        rhs.requires_grad = False
        y._update(rhs, logical_write=False)
        assert y.grad_fn_node_id == producer
        gradient, = jt.core.grad_optional(old_loss, [x], True)
        assert_device(y)
        assert_device(gradient)
        np.testing.assert_array_equal(y.numpy(), [4., 5.])
        expected = [16., 20.] if square and nonleaf else [8., 10.] if square else [6., 6.] if nonleaf else [3., 3.]
        np.testing.assert_array_equal(gradient.numpy(), expected)
        assert old_loss.item() == (20. if nonleaf and square else 5. if square else 18. if nonleaf else 9.)


@pytest.mark.parametrize('replace_source', [False, True])
@pytest.mark.parametrize('nested', [False, True])
def test_native_data_orphan_preserves_real_constructor_view_guard(replace_source, nested):
    with policy_scope(EXPLICIT_REQUIRES_GRAD):
        source = jt.array(np.array([1., 2.], dtype='float32'))
        alias = alias_of(source)
        destination = source if replace_source else alias
        rhs = jt.array(np.array([4., 5.], dtype='float32'))
        rhs.requires_grad = False
        destination._update(rhs, logical_write=False)
        target = alias_of(alias) if nested else alias
        assert_device(target)
        with pytest.raises(RuntimeError, match='view of a leaf Variable'):
            target._check_inplace_autograd()


@pytest.mark.parametrize('nonleaf', [False, True])
@pytest.mark.parametrize('operation', ['multiply', 'sin', 'exp'])
def test_native_layout_forward_preserves_logical_saved_value(nonleaf, operation):
    with policy_scope(EXPLICIT_REQUIRES_GRAD), jt.flag_scope(transpose_storage_view=1):
        x = jt.detach(jt.array(np.array([[.2, .4], [.6, .8]], dtype='float32')).transpose())
        x.requires_grad = True
        value = x * 2 if nonleaf else x
        coefficients = jt.array(np.array([[1., 2.], [3., 4.]], dtype='float32'))
        coefficients.requires_grad = False
        output = value * value if operation == 'multiply' else getattr(value, operation)()
        assert_device(output)
        saved_output = output.numpy().copy()
        loss = (output * coefficients).sum()
        replacement = jt.array(np.array([[1.1, 1.2], [1.3, 1.4]], dtype='float32'))
        replacement.requires_grad = False
        value._update(replacement, logical_write=False)
        gradients = jt.core.grad_optional(loss, [value, x] if nonleaf else [value], True)
        gradient = gradients[0]
        current = replacement.numpy()
        expected = (2 * current if operation == 'multiply' else
                    np.cos(current) if operation == 'sin' else saved_output)
        expected = expected * np.array([[1., 2.], [3., 4.]], dtype='float32')
        assert_device(gradient)
        np.testing.assert_allclose(gradient.numpy(), expected, rtol=2e-6, atol=2e-7)
        if nonleaf:
            assert_device(gradients[1])
            np.testing.assert_allclose(gradients[1].numpy(), expected * 2,
                                       rtol=2e-6, atol=2e-7)


def test_native_layout_forward_dependency_releases_its_owned_vars():
    def build_and_release():
        with policy_scope(EXPLICIT_REQUIRES_GRAD), jt.flag_scope(transpose_storage_view=1):
            x = jt.detach(jt.array(np.array([[1., 2.], [3., 4.]], dtype='float32')).transpose())
            x.requires_grad = True
            loss = (x.sin() + x * x).sum()
            gradient, = jt.core.grad_optional(loss, [x], True)
            assert_device(gradient)

    build_and_release()
    gc.collect()
    jt.sync_all(True)
    before = (jt.introspection.counters.live_vars, jt.introspection.counters.live_ops)
    for _ in range(4):
        build_and_release()
    gc.collect()
    jt.sync_all(True)
    assert (jt.introspection.counters.live_vars, jt.introspection.counters.live_ops) == before

@pytest.mark.parametrize('derived', [False, True])
@pytest.mark.parametrize('strided', [False, True])
def test_native_store_preserves_unmaterialized_saved_output(derived, strided):
    with policy_scope(EXPLICIT_REQUIRES_GRAD), jt.flag_scope(transpose_storage_view=1):
        source = jt.array(np.zeros((2, 2), dtype='float32'))
        if strided:
            source = jt.detach(source.transpose())
        source.requires_grad = True
        value = source + 1 if derived else source
        loss = value.exp().sum()
        # A requested scalar can run without allocating the exp needed by AD.
        loss.sync(False, False)
        unrelated = jt.array(np.ones((2, 2), dtype='float32')).exp()
        assert not unrelated._mem_ptr_now
        rhs = jt.array(np.full((2, 2), .5, dtype='float32')).stop_grad()
        with jt.no_grad():
            source._copy_into(rhs)
        # Preserve only readers of this allocation, not unrelated graph roots.
        assert not unrelated._mem_ptr_now
        gradient, = jt.core.grad_optional(loss, [source], True)
        assert_device(source)
        assert_device(gradient)
        np.testing.assert_allclose(gradient.numpy(), np.full((2, 2), np.e if derived else 1),
                                   rtol=2e-6, atol=2e-7)
        np.testing.assert_array_equal(source.numpy(), np.full((2, 2), .5))
