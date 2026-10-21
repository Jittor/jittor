"""StepLR checkpoint metadata and recurrence match the pinned Torch 2.7 oracle."""

import copy

import numpy as np
import pytest
import torch

from _release_test_helpers import public_device_scope, step_scheduler
from _scheduler_checkpoint_helpers import clone, optimizer, snapshot, update


@pytest.fixture
def steplr_device():
    with public_device_scope() as name:
        yield name


def test_default_named_groups_and_modern_checkpoint_schema(steplr_device):
    opt = optimizer(steplr_device)
    schedule = torch.optim.lr_scheduler.StepLR(opt, 2, gamma=.5)
    assert [group["initial_lr"] for group in opt.param_groups] == [.2, .4]
    assert schedule.state_dict() == {
        "step_size": 2, "gamma": .5, "base_lrs": [.2, .4], "last_epoch": 0,
        "_step_count": 1, "_get_lr_called_within_step": False, "_last_lr": [.2, .4]}


def test_tensor_initial_lr_is_cloned_and_preserves_existing_value(steplr_device):
    opt = optimizer(steplr_device)
    rate = torch.tensor(.2, device=steplr_device)
    opt.param_groups[0]["lr"] = rate
    opt.param_groups[1]["initial_lr"] = .7
    schedule = torch.optim.lr_scheduler.StepLR(opt, 2, gamma=.5)
    initial = opt.param_groups[0]["initial_lr"]
    assert initial is not rate
    rate.fill_(.9)
    np.testing.assert_allclose(initial.detach().cpu().numpy(), .2, atol=1e-7, rtol=0)
    assert schedule.base_lrs[1] == .7


@pytest.mark.parametrize("last_epoch", [-2, 0, 3])
def test_nondefault_epoch_rejects_missing_group_initial_lr(steplr_device, last_epoch):
    opt = optimizer(steplr_device)
    opt.param_groups[0]["initial_lr"] = .2
    before = [group["lr"] for group in opt.param_groups]
    with pytest.raises(KeyError, match=r"initial_lr.*param_groups\[1\]"):
        torch.optim.lr_scheduler.StepLR(opt, 2, gamma=.5, last_epoch=last_epoch)
    assert [group["lr"] for group in opt.param_groups] == before


@pytest.mark.parametrize("last_epoch,expected", [
    (-2, [.3, .7]), (0, [.3, .7]), (3, [.15, .35])])
def test_constructor_uses_initial_base_and_current_chainable_lr(steplr_device, last_epoch, expected):
    opt = optimizer(steplr_device)
    for group, base, current in zip(opt.param_groups, [.2, .4], [.3, .7]):
        group["initial_lr"] = base
        group["lr"] = current
    schedule = torch.optim.lr_scheduler.StepLR(opt, 2, gamma=.5, last_epoch=last_epoch)
    assert schedule.base_lrs == [.2, .4]
    assert schedule.get_last_lr() == expected
    assert [group["lr"] for group in opt.param_groups] == expected


def test_chainable_step_preserves_external_lr_and_epoch_uses_closed_form(steplr_device):
    opt = optimizer(steplr_device)
    schedule = torch.optim.lr_scheduler.StepLR(opt, 2, gamma=.5)
    update(opt, schedule)
    for group, value in zip(opt.param_groups, [.3, .7]):
        group["lr"] = value
    update(opt, schedule)
    assert schedule.get_last_lr() == [.15, .35]
    step_scheduler(opt, schedule, epoch=5)
    assert schedule.get_last_lr() == [.05, .1]


def test_eight_updates_full_state_matches_fresh_three_plus_five_restore(steplr_device):
    opt = optimizer(steplr_device)
    schedule = torch.optim.lr_scheduler.StepLR(opt, 1, gamma=.8)
    records = []
    for index in range(8):
        update(opt, schedule)
        records.append(snapshot(opt, schedule))
        if index == 2:
            saved_optimizer = clone(opt.state_dict())
            saved_scheduler = copy.deepcopy(schedule.state_dict())
            saved_parameters = [parameter.detach().clone() for group in opt.param_groups
                                for parameter in group["params"]]
    restored = optimizer(steplr_device)
    restored_schedule = torch.optim.lr_scheduler.StepLR(restored, 1, gamma=.8)
    restored.load_state_dict(saved_optimizer)
    restored_schedule.load_state_dict(saved_scheduler)
    with torch.no_grad():
        for parameter, saved in zip(
                [p for group in restored.param_groups for p in group["params"]], saved_parameters):
            parameter.copy_(saved)
    assert snapshot(restored, restored_schedule) == records[2]
    for expected in records[3:]:
        update(restored, restored_schedule)
        assert snapshot(restored, restored_schedule) == expected


def test_get_lr_warns_outside_scheduler_step(steplr_device):
    opt = optimizer(steplr_device)
    schedule = torch.optim.lr_scheduler.StepLR(opt, 2, gamma=.5)
    with pytest.warns(UserWarning, match="get_last_lr"):
        schedule.get_lr()
