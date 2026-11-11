"""Batched optimizer stores retain native writes and decline before mutation."""
import numpy as np
import jittor as jt
import torch

from _helpers.device_types import instantiate_device_type_tests
from jittor.compat.torch import optimizer_adam
from jittor.compat.torch.installers.tensor.method_api import _optimizer_update


def _arrays(tensors):
    return [x.detach().cpu().numpy().copy() for x in tensors]


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
        native = jt.core._try_optimizer_copies
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
        with patch.object(jt.core, '_try_optimizer_copies', observe):
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
        with patch.object(jt.core, '_try_optimizer_copies', side_effect=AssertionError('must decline')):
            assert not optimizer_adam._try_batch_update(active, updates)
            p = torch.nn.Parameter(torch.ones(2, device=device))
            active[0] = (p, m, v, value, 0)
            for name in ('active', 'tracing'):
                with patch.object(step_capture, name, return_value=True):
                    assert not optimizer_adam._try_batch_update(active, updates)


instantiate_device_type_tests(TestOptimizerBatchWriteback, globals(), only_for=('cpu', 'cuda'))
