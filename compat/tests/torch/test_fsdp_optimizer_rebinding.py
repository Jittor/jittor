"""FSDP view rotation preserves Adam state without retaining old parameters.

These local tests exercise the metadata/cache boundary without collectives;
``test_fsdp2_nccl`` verifies the corresponding real multi-rank training path.
"""

import weakref
import gc
import os
from unittest import mock

import pytest

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
    def test_adam_initializers_record_parameter_identity_before_first_step(self, device):
        for algorithm in (torch.optim.Adam, torch.optim.AdamW):
            for foreach in (None, False):
                first = _parameter([1., 2.], device)
                second = _parameter([3.], device)
                optimizer = algorithm([{"params": [first]}, {"params": [second]}],
                                      lr=.1, foreach=foreach)
                for group, parameter in zip(optimizer.param_groups, (first, second)):
                    assert len(group["_torch_adam_params"]) == 1
                    assert group["_torch_adam_params"][0] is parameter
                    assert group["foreach"] is foreach
                assert optimizer.n_step == 0

    def test_owner_enumeration_releases_parameter_without_cyclic_gc(self, device):
        from jittor.compat.fsdp2.shard import _named_parameters_with_owner

        model = torch.nn.Linear(2, 1, bias=False).to(device)
        marker = StateRecord()
        object.__setattr__(model.weight, "_fsdp_cache_lifetime_probe", marker)
        lifetime = weakref.ref(marker)
        del marker
        was_enabled = gc.isenabled()
        gc.disable()
        try:
            parameters = _named_parameters_with_owner(model)
            assert parameters[0][0] == "weight" and parameters[0][3] is model.weight
            model.weight = torch.nn.Parameter(torch.tensor([[3., 4.]], device=device))
            del parameters
            assert lifetime() is None, "owner enumeration retained a superseded Parameter"
        finally:
            if was_enabled:
                gc.enable()

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


    def test_public_singleton_rebinds_trained_adam_slots_and_continues(self, device):
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import fully_shard

        for algorithm in (torch.optim.Adam, torch.optim.AdamW):
            for flat in (False, True):
                model = torch.nn.Linear(2, 2).to(device)
                with torch.no_grad():
                    model.weight.copy_(torch.tensor([[.5, -.25], [.125, .375]], device=device))
                    model.bias.copy_(torch.tensor([.125, -.125], device=device))
                optimizer = algorithm([
                    {"params": [model.weight], "lr": .02},
                    {"params": [model.bias], "lr": .03},
                ], betas=(.6, .8), eps=.005, weight_decay=.02, foreach=False)
                originals = dict(model.named_parameters())
                model.weight.grad = torch.tensor([[.25, -.5], [.75, -.125]], device=device)
                model.bias.grad = torch.tensor([.5, -.25], device=device)
                optimizer.step()
                before = {name: value.detach().cpu().numpy().copy() for name, value in originals.items()}
                views = {name: optimizer.state[value] for name, value in originals.items()}
                moments = {name: {key: entry[key].detach().cpu().numpy().copy()
                                  for key in ("exp_avg", "exp_avg_sq")}
                           for name, entry in views.items()}
                lifetime = []
                for value in originals.values():
                    marker = StateRecord()
                    object.__setattr__(value, "_fsdp_cache_lifetime_probe", marker)
                    lifetime.append(weakref.ref(marker))
                del marker, value
                containers = [group["params"] for group in optimizer.param_groups]
                with mock.patch.dict(os.environ, {"JITTOR_FSDP2_FLAT": "1" if flat else "0"}):
                    fully_shard(model, mesh=init_device_mesh(device, (1,)))
                current = dict(model.named_parameters())
                assert set(current) == set(originals)
                for group, container in zip(optimizer.param_groups, containers):
                    assert group["params"] is container
                for index, (name, value) in enumerate(current.items()):
                    assert optimizer.param_groups[index]["params"][0] is value
                    assert value is not originals[name]
                    assert optimizer.state[value] is views[name]
                    assert float(views[name]["step"]) == 1
                    for key in ("exp_avg", "exp_avg_sq"):
                        slot = views[name][key]
                        assert tuple(slot.shape) == tuple(value.shape)
                        assert slot.dtype == value.dtype
                        np.testing.assert_array_equal(slot.detach().cpu().numpy().reshape(-1), moments[name][key].reshape(-1))
                originals.clear()
                assert all(ref() is None for ref in lifetime), "optimizer/state view retained an original full Parameter"
                optimizer.zero_grad(set_to_none=True)
                x = torch.tensor([[.25, -.5], [.75, .125]], device=device, requires_grad=True)
                loss = model(x).square().mean()
                loss.backward()
                gradients = {name: value.grad.full_tensor().detach().cpu().numpy().copy()
                             for name, value in model.named_parameters()}
                optimizer.step()
                current = dict(model.named_parameters())
                for index, (name, value) in enumerate(current.items()):
                    gradient = gradients[name]
                    start = before[name]
                    if algorithm is torch.optim.Adam:
                        gradient = gradient + .02 * start
                        decayed = start
                    else:
                        decayed = start * (1 - optimizer.param_groups[index]["lr"] * .02)
                    expected_m = .6 * moments[name]["exp_avg"] + .4 * gradient
                    expected_v = .8 * moments[name]["exp_avg_sq"] + .2 * gradient * gradient
                    expected = decayed - optimizer.param_groups[index]["lr"] * (expected_m / (1 - .6 ** 2)) / (
                        np.sqrt(expected_v / (1 - .8 ** 2)) + .005)
                    np.testing.assert_allclose(value.full_tensor().detach().cpu().numpy(), expected, rtol=2e-6, atol=2e-6)
                    state = optimizer.state[value]
                    assert state is views[name]
                    assert float(state["step"]) == 2
                    np.testing.assert_allclose(state["exp_avg"].detach().cpu().numpy().reshape(-1), expected_m.reshape(-1), rtol=2e-6, atol=2e-6)
                    np.testing.assert_allclose(state["exp_avg_sq"].detach().cpu().numpy().reshape(-1), expected_v.reshape(-1), rtol=2e-6, atol=2e-6)

    def test_full_slot_partition_uses_real_entry_intervals(self, device):
        from jittor.compat.fsdp2.optimizer import _localize_optimizer_buffer

        # Flat entries can span ranks or have no local elements; nonflat shards
        # retain padded storage, including a completely padded rank.
        cases = [
            (False, 1, 0, 4, 0, 4, [1., 2., 3., 4.]),
            (False, 2, 0, 5, 0, 3, [1., 2., 3.]),
            (False, 2, 1, 5, 0, 3, [4., 5., 0.]),
            (False, 2, 1, 1, 0, 1, [0.]),
            (True, 2, 0, 4, 2, 3, [1.]),
            (True, 2, 1, 4, 2, 3, [2., 3., 4.]),
            (True, 2, 0, 2, 4, 3, []),
            (True, 2, 1, 2, 4, 3, [1., 2.]),
        ]
        for flat, world, rank, count, offset, span, expected in cases:
            full = torch.tensor(np.arange(1, count + 1, dtype=np.float32), device=device)
            old = full.detach().clone().requires_grad_(True)
            object.__setattr__(old, "_jittor_fsdp2_role", "retired")
            local = torch.zeros(len(expected), device=device, requires_grad=True)
            entry = StateRecord(shard=local, numel=count, flat_offset=offset,
                                shard_numel=span, padded_numel=span * world)
            state = StateRecord(true_fsdp_flat=flat, true_fsdp_world_size=world,
                                true_fsdp_rank=rank, true_fsdp_flat_shard_numel=span)
            actual = _localize_optimizer_buffer(full, old, state, entry)
            assert tuple(actual.shape) == tuple(local.shape)
            assert actual.dtype == full.dtype and actual.device.type == device
            assert not actual.requires_grad
            np.testing.assert_array_equal(actual.detach().cpu().numpy(), expected)

    def test_invalid_full_slot_is_rejected_before_parameter_rebinding(self, device):
        old = _parameter([1., 2., 3.], device)
        optimizer = torch.optim.Adam([old], lr=.1, foreach=False)
        old.grad = torch.tensor([1., 2., 3.], device=device)
        optimizer.step()
        cached = optimizer.state[old]
        group = optimizer.param_groups[0]
        new = _parameter([1., 2., 3.], device)
        state = StateRecord(true_fsdp_initialized=True, true_fsdp_world_size=1,
                            true_fsdp_rank=0, true_fsdp_flat=False)
        entry = StateRecord(shard=new, numel=3, shard_numel=3, padded_numel=3)
        object.__setattr__(old, "_jittor_fsdp2_state", state)
        object.__setattr__(old, "_jittor_fsdp2_entry", entry)
        object.__setattr__(old, "_jittor_fsdp2_role", "retired")
        group["m"][0] = torch.tensor([4., 5.], device=device)
        with pytest.raises(ValueError, match="full buffer does not match its parameter size"):
            refresh_optimizer_fsdp_params(optimizer)
        assert group["params"][0] is old
        assert optimizer.state[old] is cached
        assert float(cached["step"]) == 1
        np.testing.assert_array_equal(group["m"][0].detach().cpu().numpy(), [4., 5.])

    def test_new_half_slot_preserves_fp32_master_state_and_parameter_device(self, device):
        for values in ([1., -2.], []):
            parameter = torch.tensor(values, device=device, dtype=torch.bfloat16, requires_grad=True)
            group = {"params": [parameter], "_torch_adam_params": (),
                     "m": [], "values": [], "_torch_steps": []}
            _sync_adam_group_params(group)
            assert len(group["_torch_adam_params"]) == 1 and group["_torch_adam_params"][0] is parameter
            assert group["_torch_steps"] == [0]
            for key in ("m", "values"):
                buffer = group[key][0]
                assert buffer.dtype == torch.float32 and buffer.device.type == device
                assert parameter.dtype == torch.bfloat16
                assert tuple(buffer.shape) == tuple(parameter.shape)
                np.testing.assert_array_equal(buffer.detach().float().cpu().numpy(), np.zeros(len(values)))


instantiate_device_type_tests(TestFsdpOptimizerRebinding, globals(), only_for=("cpu", "cuda"))
