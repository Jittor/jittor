"""LinearLR public trajectories frozen against independent PyTorch 2.6.

This file runs unchanged against PyTorch and the Torch shim. Scheduler LR
arithmetic is Python float state; tensor placement is checked separately.
"""

import copy
import warnings

import pytest
import torch

from _release_test_helpers import (
    linear_schedule as make_schedule, optimizer_lrs as lrs,
    public_device_scope, step_scheduler as step,
)


@pytest.fixture(autouse=True)
def linear_lr_device():
    with public_device_scope() as name:
        yield name


def test_get_lr_uses_live_groups_without_mutating_last_lr(linear_lr_device):
    optimizer, scheduler = make_schedule(linear_lr_device)
    assert lrs(optimizer) == [0.1, 0.2]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        assert scheduler.get_lr() == [0.05, 0.1]
    assert scheduler.get_last_lr() == [0.1, 0.2]
    assert lrs(optimizer) == [0.1, 0.2]


def test_first_step_preserves_manual_group_lr_changes(linear_lr_device):
    optimizer, scheduler = make_schedule(linear_lr_device)
    optimizer.param_groups[0]["lr"] = 0.75
    optimizer.param_groups[1]["lr"] = 0.125
    step(optimizer, scheduler)
    assert lrs(optimizer) == [0.9375, 0.15625]
    assert scheduler.get_last_lr() == lrs(optimizer)


def test_completed_schedule_keeps_manual_lr_changes(linear_lr_device):
    optimizer, scheduler = make_schedule(linear_lr_device)
    for _ in range(4):
        step(optimizer, scheduler)
    optimizer.param_groups[0]["lr"] = 0.75
    optimizer.param_groups[1]["lr"] = 0.125
    step(optimizer, scheduler)
    assert lrs(optimizer) == [0.75, 0.125]
    assert scheduler.get_last_lr() == [0.75, 0.125]


def test_explicit_epoch_uses_base_lrs_after_manual_change(linear_lr_device):
    optimizer, scheduler = make_schedule(linear_lr_device)
    optimizer.param_groups[0]["lr"] = 0.75
    optimizer.param_groups[1]["lr"] = 0.125
    step(optimizer, scheduler, epoch=2)
    assert scheduler.last_epoch == 2
    assert lrs(optimizer) == [0.15000000000000002, 0.30000000000000004]


def test_restore_keeps_live_lr_for_next_step(linear_lr_device):
    optimizer, scheduler = make_schedule(linear_lr_device)
    optimizer.param_groups[0]["lr"] = 0.75
    optimizer.param_groups[1]["lr"] = 0.125
    saved_optimizer = copy.deepcopy(optimizer.state_dict())
    saved_scheduler = copy.deepcopy(scheduler.state_dict())
    restored_optimizer, restored_scheduler = make_schedule(linear_lr_device)
    restored_optimizer.load_state_dict(saved_optimizer)
    restored_scheduler.load_state_dict(saved_scheduler)
    step(optimizer, scheduler)
    step(restored_optimizer, restored_scheduler)
    assert lrs(restored_optimizer) == lrs(optimizer) == [0.9375, 0.15625]
    assert restored_scheduler.state_dict() == scheduler.state_dict()
    assert restored_scheduler.get_last_lr() == scheduler.get_last_lr()


def test_zero_total_iters_applies_start_factor_then_keeps_live_lr(linear_lr_device):
    optimizer, scheduler = make_schedule(linear_lr_device, total_iters=0)
    assert lrs(optimizer) == [0.1, 0.2]
    step(optimizer, scheduler)
    assert lrs(optimizer) == [0.1, 0.2]


def test_zero_end_factor_does_not_reset_a_later_manual_change(linear_lr_device):
    optimizer, scheduler = make_schedule(linear_lr_device, end_factor=0.0, total_iters=2)
    step(optimizer, scheduler)
    assert lrs(optimizer) == [0.05, 0.1]
    step(optimizer, scheduler)
    assert lrs(optimizer) == [0.0, 0.0]
    optimizer.param_groups[0]["lr"] = 0.75
    step(optimizer, scheduler)
    assert lrs(optimizer) == [0.75, 0.0]


@pytest.mark.parametrize("options", [
    {"start_factor": 0.0}, {"start_factor": -0.1}, {"start_factor": 1.01},
    {"end_factor": -0.1}, {"end_factor": 1.01},
])
def test_factor_range_is_validated_before_initial_step(linear_lr_device, options):
    with pytest.raises(ValueError):
        make_schedule(linear_lr_device, **options)


def test_vision_decay_has_exact_reference_float_trajectory(linear_lr_device):
    # Values from independent torch==2.6.0+cu124; strict scheduler state uses
    # these float values, including the accumulated final rounding.
    expected = {0: 0.01, 1: 0.00991, 2: 0.00982, 3: 0.00973,
                50: 0.005500000000000003, 99: 0.0010900000000000005,
                100: 0.0010000000000000005, 101: 0.0010000000000000005}
    parameter = torch.nn.Parameter(torch.tensor([1.0], device=linear_lr_device))
    assert parameter.device.type == linear_lr_device
    optimizer = torch.optim.SGD([parameter], lr=0.01)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.1, total_iters=100)
    for update in range(102):
        if update in expected:
            assert scheduler.get_last_lr() == [expected[update]], update
            assert lrs(optimizer) == [expected[update]], update
        if update < 101:
            step(optimizer, scheduler)
