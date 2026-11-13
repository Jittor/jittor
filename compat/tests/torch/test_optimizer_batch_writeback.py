"""Batched optimizer stores retain native writes and decline before mutation."""
import numpy as np
import jittor as jt
import torch

from _helpers.device_types import instantiate_device_type_tests
from jittor.compat.torch import optimizer_adam
from jittor.compat.torch.installers.tensor import method_api as tensor_methods
from jittor.compat.torch.installers.tensor.method_api import _optimizer_update


def _arrays(tensors):
    return [x.detach().cpu().numpy().copy() for x in tensors]


def _packing_case():
    parameter = torch.nn.Parameter(torch.tensor([1., 2.], device='cpu'))
    moment, variance, gradient = [torch.zeros(2, device='cpu') for _ in range(3)]
    values = tuple(torch.ones(2, device='cpu') for _ in range(3))
    active = [(parameter, moment, variance, gradient, 0)]
    frontend = optimizer_adam.get_install_context(jt).target_namespace
    return active, [values], (jt.Var, frontend.Tensor, frontend.nn.Parameter)


def test_packed_updates_keep_identity_order_and_live_replacements():
    active, updates, types = _packing_case()
    first = jt.core._try_pack_optimizer_updates(
        active, updates, types, _optimizer_update, optimizer_adam._BATCH_UPDATE_PROTOCOL)
    assert first is not None
    assert all(a is b for a, b in zip(first[0], active[0][:3]))
    assert all(a is b for a, b in zip(first[1], updates[0]))
    replacement = torch.nn.Parameter(torch.tensor([3., 4.], device='cpu'))
    active[0] = (replacement, *active[0][1:])
    updates[0] = tuple(torch.full((2,), 5., device='cpu') for _ in range(3))
    second = jt.core._try_pack_optimizer_updates(
        active, updates, types, _optimizer_update, optimizer_adam._BATCH_UPDATE_PROTOCOL)
    assert second is not None and second[0][0] is replacement
    assert all(a is b for a, b in zip(second[1], updates[0]))
    assert first[0][0] is not replacement


def test_update_iterators_are_left_for_the_ordinary_writer():
    active, updates, _ = _packing_case()
    producer = iter(updates)
    assert not optimizer_adam._try_batch_update(active, producer)
    assert next(producer) is updates[0]


def test_packing_preserves_custom_global_getters():
    from unittest.mock import patch

    active, updates, _ = _packing_case()
    calls = []

    def observed(owner, name, default=None):
        calls.append((owner, name))
        return getattr(owner, name, default)

    with patch.object(optimizer_adam, 'getattr', observed, create=True):
        with tensor_methods._optimizer_write_scope():
            optimizer_adam._try_batch_update(active, updates)
    assert calls == [(type(target), '_optimizer_update') for target in active[0][:3]]


def test_writer_descriptors_are_observed_only_by_python_fallback():
    from unittest.mock import patch

    calls = []

    class Meta(type):
        @property
        def _optimizer_update(cls):
            calls.append(cls)
            return _optimizer_update

    class Target(metaclass=Meta):
        _optimizer_update = _optimizer_update

    target = Target()
    assert jt.core._try_optimizer_writers(
        [target], _optimizer_update, tensor_methods._BATCH_WRITE_PROTOCOL) is None
    assert not calls
    with patch.object(jt.core, '_commit_shared_updates', return_value=False):
        with tensor_methods._optimizer_write_scope():
            assert not tensor_methods._optimizer_update_many([target], [target])
    assert calls == [Target]


class TestOptimizerBatchWriteback:
    def test_dense_stores_preserve_storage_aliases_and_versions(self, device):
        targets = [torch.tensor([1., 2.], device=device) for _ in range(3)]
        values = [torch.tensor([3., 4.], device=device) for _ in targets]
        aliases = [x.detach() for x in targets]
        pointers = [x.data_ptr() for x in targets]
        versions = [x._autograd_version() for x in targets]
        jt.sync_all()
        with torch.no_grad():
            accepted = jt.core._try_optimizer_copies(targets, values)
        assert accepted == (device == 'cuda')
        for target, alias, pointer, version in zip(targets, aliases, pointers, versions):
            assert target.data_ptr() == alias.data_ptr() == pointer
            assert target._autograd_version() == alias._autograd_version() == version + int(accepted)
            np.testing.assert_array_equal(alias.cpu().numpy(), [3., 4.] if accepted else [1., 2.])

    def test_invalid_last_pair_declines_without_writing_first(self, device):
        for reason in ('shape', 'dtype', 'view', 'pending', 'not_tensor'):
            first = torch.tensor([1., 2.], device=device)
            last = torch.tensor([5., 6.], device=device)
            good = torch.tensor([3., 4.], device=device)
            if reason == 'shape':
                bad = torch.ones(3, device=device)
            elif reason == 'dtype':
                bad = torch.ones(2, device=device, dtype=torch.float64)
            elif reason == 'view':
                bad = torch.ones(4, device=device)[:2]
            else:
                bad = torch.ones(2, device=device)
            jt.sync_all()
            if reason == 'pending':
                bad = bad + 1
            elif reason == 'not_tensor':
                bad = object()
            version = first._autograd_version()
            with torch.no_grad():
                assert not jt.core._try_optimizer_copies([first, last], [good, bad]), reason
            assert first._autograd_version() == version
            np.testing.assert_array_equal(first.cpu().numpy(), [1., 2.])

    def test_grad_recording_declines_without_mutation(self, device):
        target = torch.ones(2, device=device, requires_grad=True)
        value = torch.zeros(2, device=device)
        jt.sync_all()
        with torch.enable_grad():
            assert not jt.core._try_optimizer_copies([target], [value])
        np.testing.assert_array_equal(target.detach().cpu().numpy(), [1., 1.])

    def test_adamw_uses_batch_and_preserves_custom_writer(self, device):
        from unittest.mock import patch
        native = optimizer_adam._optimizer_update_many
        outcomes, writes = [], []

        def observe(*args):
            accepted = native(*args)
            outcomes.append(accepted)
            return accepted

        def custom(target, value):
            writes.append(target)
            return _optimizer_update(target, value)

        parameter = torch.nn.Parameter(torch.tensor([1., 2.], device=device))
        optimizer = torch.optim.AdamW([parameter], lr=.125, weight_decay=0.)
        with patch.object(optimizer_adam, '_optimizer_update_many', observe):
            parameter.grad = torch.ones_like(parameter)
            optimizer.step()
            if device == 'cuda':
                assert True in outcomes
            before = _arrays([parameter])[0]
            with patch.object(torch.nn.Parameter, '_optimizer_update', custom):
                parameter.grad = torch.ones_like(parameter)
                optimizer.step()
        assert len(writes) == 1 and writes[0] is parameter
        np.testing.assert_allclose(_arrays([parameter])[0], before - .125, atol=1e-6)

    def test_subclasses_and_capture_decline(self, device):
        from unittest.mock import patch
        from jittor._runtime import step_capture

        class CustomParameter(torch.nn.Parameter):
            pass

        p = CustomParameter(torch.ones(2, device=device))
        m, v, value = [torch.zeros(2, device=device) for _ in range(3)]
        active = [(p, m, v, value, 0)]
        updates = [(value, value, value)]
        with patch.object(optimizer_adam, '_optimizer_update_many', side_effect=AssertionError('must decline')):
            assert not optimizer_adam._try_batch_update(active, updates)
            p = torch.nn.Parameter(torch.ones(2, device=device))
            active[0] = (p, m, v, value, 0)
            for name in ('active', 'tracing'):
                with patch.object(step_capture, name, return_value=True):
                    assert not optimizer_adam._try_batch_update(active, updates)


instantiate_device_type_tests(TestOptimizerBatchWriteback, globals(), only_for=('cpu', 'cuda'))
