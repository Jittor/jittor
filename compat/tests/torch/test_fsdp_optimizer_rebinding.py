"""FSDP view rotation preserves Adam state without retaining old parameters.

These local tests exercise the metadata/cache boundary without collectives;
``test_fsdp2_nccl`` verifies the corresponding real multi-rank training path.
"""

import weakref

import numpy as np
import torch

from _helpers.device_types import instantiate_device_type_tests
from jittor.compat.fsdp2.common import StateRecord
from jittor.compat.fsdp2.optimizer import refresh_optimizer_fsdp_params
from jittor.compat.torch.optimizer_api import _sync_adam_group_params


def _parameter(values, device):
    return torch.tensor(values, device=device, requires_grad=True)


def _rotation(old):
    new = old.detach().clone().requires_grad_(True)
    state = StateRecord(true_fsdp_initialized=True)
    entry = StateRecord(shard=new)
    for parameter in (old, new):
        object.__setattr__(parameter, "_jittor_fsdp2_state", state)
        object.__setattr__(parameter, "_jittor_fsdp2_entry", entry)
    marker = StateRecord()
    object.__setattr__(old, "_fsdp_cache_lifetime_probe", marker)
    return new, state, weakref.ref(marker)


class TestFsdpOptimizerRebinding:
    def test_empty_checkpoint_resolves_names_after_initialization_rotates_views(self, device):
        from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict

        model = torch.nn.Linear(2, 1, bias=False).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.1, foreach=False)
        before = model.weight.detach().cpu().numpy().copy()
        step = optimizer.step
        rotations = []

        def rotate_views():
            result = step()
            group = optimizer.param_groups[0]
            old = group['params'][0]
            new = torch.nn.Parameter(old.detach().clone(), requires_grad=True)
            group['params'][0] = new
            _sync_adam_group_params(group, replacements=((old, new),))
            model.weight = new
            rotations.append((id(old), id(new)))
            return result

        # Reproduce FSDP's real optimizer-owned public view rotation locally;
        # the maintained two-rank fresh-process gate exercises actual FSDP.
        optimizer.step = rotate_views
        checkpoint = get_optimizer_state_dict(model, optimizer)
        assert len(rotations) == 1 and rotations[0][0] != rotations[0][1]
        assert checkpoint['param_groups'][0]['params'] == ['weight']
        assert checkpoint['param_groups'][0]['lr'] == .1
        assert set(checkpoint['state']) == {'weight'}
        entry = checkpoint['state']['weight']
        assert float(entry['step'].item()) == 1
        for field in ('exp_avg', 'exp_avg_sq'):
            np.testing.assert_array_equal(entry[field].detach().cpu().numpy(), np.zeros_like(before))
        np.testing.assert_array_equal(model.weight.detach().cpu().numpy(), before)
        assert model.weight.grad is None
        assert optimizer.param_groups[0]['params'][0] is model.weight
        get_optimizer_state_dict(model, optimizer)
        assert len(rotations) == 1

    def test_refresh_releases_old_view_and_preserves_state(self, device):
        for algorithm in (torch.optim.Adam, torch.optim.AdamW):
            old = _parameter([1., 2.], device)
            optimizer = algorithm([old], lr=.1, weight_decay=0, foreach=False)
            old.grad = torch.tensor([1., -2.], device=device)
            optimizer.step()
            group = optimizer.param_groups[0]
            moment, variance = group["m"][0], group["values"][0]
            new, state, lifetime = _rotation(old)
            refresh_optimizer_fsdp_params(optimizer, {id(state)})
            del old
            assert lifetime() is None, "the optimizer retained a superseded shard"
            assert group["params"][0] is new
            assert group["_torch_adam_params"][0] is new
            assert group["m"][0] is moment
            assert group["values"][0] is variance
            assert group["_torch_steps"] == [1]

    def test_rotation_and_user_reorder_keep_logical_moments(self, device):
        first = _parameter([1., 2.], device)
        second = _parameter([3., 4., 5.], device)
        optimizer = torch.optim.Adam([first, second], lr=.1, foreach=False)
        first.grad = torch.tensor([1., 2.], device=device)
        second.grad = torch.tensor([3., 4., 5.], device=device)
        optimizer.step()
        group = optimizer.param_groups[0]
        first_moment, second_moment = group["m"]
        first_variance, second_variance = group["values"]
        rotated, state, lifetime = _rotation(first)
        fresh = _parameter([6.], device)
        group["params"] = [second, first, fresh]
        refresh_optimizer_fsdp_params(optimizer, {id(state)})
        del first
        assert lifetime() is None
        assert group["params"][1] is rotated
        assert group["m"][0] is second_moment
        assert group["m"][1] is first_moment
        assert group["values"][0] is second_variance
        assert group["values"][1] is first_variance
        assert group["_torch_steps"] == [1, 1, 0]
        np.testing.assert_array_equal(group["m"][2].cpu().numpy(), [0.])
        np.testing.assert_array_equal(group["values"][2].cpu().numpy(), [0.])

    def test_unselected_fsdp_state_remains_untouched(self, device):
        old = _parameter([1., 2.], device)
        optimizer = torch.optim.Adam([old], lr=.1, foreach=False)
        old.grad = torch.tensor([1., 2.], device=device)
        optimizer.step()
        group = optimizer.param_groups[0]
        new, state, lifetime = _rotation(old)
        refresh_optimizer_fsdp_params(optimizer, set())
        assert group["params"][0] is old
        assert group["_torch_adam_params"][0] is old
        assert group["_torch_steps"] == [1]
        refresh_optimizer_fsdp_params(optimizer, {id(state)})
        del old
        assert lifetime() is None
        assert group["params"][0] is new
        assert group["_torch_steps"] == [1]

    def test_rebinding_owner_continues_two_actual_adam_updates(self, device):
        for algorithm in (torch.optim.Adam, torch.optim.AdamW):
            old = _parameter([1., 2.], device)
            optimizer = algorithm([old], lr=.1, weight_decay=0, foreach=False)
            old.grad = torch.tensor([1., -2.], device=device)
            optimizer.step()
            new = old.detach().clone().requires_grad_(True)
            group = optimizer.param_groups[0]
            group["params"] = [new]
            _sync_adam_group_params(group, replacements=((old, new),))
            new.grad = torch.tensor([2., -4.], device=device)
            optimizer.step()
            np.testing.assert_allclose(group["m"][0].cpu().numpy(), [.29, -.58], rtol=1e-6)
            np.testing.assert_allclose(group["values"][0].cpu().numpy(), [.004999, .019996], rtol=1e-6)
            assert group["_torch_steps"] == [2]
            expected = np.array([.9, 2.1]) - .1 * np.array([.29, -.58]) / (1 - .9 ** 2) / (
                np.sqrt([.004999, .019996]) / np.sqrt(1 - .999 ** 2) + 1e-8)
            np.testing.assert_allclose(new.detach().cpu().numpy(), expected, rtol=1e-6)


instantiate_device_type_tests(TestFsdpOptimizerRebinding, globals(), only_for=("cpu", "cuda"))
