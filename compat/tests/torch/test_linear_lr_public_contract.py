"""LinearLR public trajectories frozen against independent PyTorch 2.6.

This file runs unchanged against PyTorch and the Torch shim. Scheduler LR
arithmetic is Python float state; tensor placement is checked separately.
"""

import copy
import os
import warnings

import pytest
import torch


@pytest.fixture(autouse=True)
def device_scope():
    device = os.environ.get("JITTOR_TEST_DEVICES", "cpu").split(",")[0]
    assert device in ("cpu", "cuda")
    if device == "cuda":
        assert torch.cuda.is_available(), "requires a real CUDA device"
    if hasattr(torch, "_torch_compat_install_context"):
        import jittor as jt
        from jittor._runtime.fallback import forbid_backend_fallbacks
        with jt.runtime.scope(use_cuda=int(device == "cuda"), backend_fallback="error"):
            with forbid_backend_fallbacks():
                before = jt.core.backend_fallback_count()
                yield device
                assert jt.core.backend_fallback_count() == before
    else:
        yield device


def make_schedule(device, *, start_factor=0.5, end_factor=1.0, total_iters=4):
    parameters = [torch.nn.Parameter(torch.tensor([1.0], device=device)),
                  torch.nn.Parameter(torch.tensor([2.0], device=device))]
    assert all(parameter.device.type == device for parameter in parameters)
    optimizer = torch.optim.SGD([{"params": [parameters[0]], "lr": 0.2},
                                 {"params": [parameters[1]], "lr": 0.4}])
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=start_factor, end_factor=end_factor,
        total_iters=total_iters)
    return optimizer, scheduler


def step(optimizer, scheduler, epoch=None):
    optimizer.step()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        scheduler.step(epoch)


def lrs(optimizer):
    return [group["lr"] for group in optimizer.param_groups]


def test_get_lr_uses_live_groups_without_mutating_last_lr(device_scope):
    optimizer, scheduler = make_schedule(device_scope)
    assert lrs(optimizer) == [0.1, 0.2]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        assert scheduler.get_lr() == [0.05, 0.1]
    assert scheduler.get_last_lr() == [0.1, 0.2]
    assert lrs(optimizer) == [0.1, 0.2]


def test_first_step_preserves_manual_group_lr_changes(device_scope):
    optimizer, scheduler = make_schedule(device_scope)
    optimizer.param_groups[0]["lr"] = 0.75
    optimizer.param_groups[1]["lr"] = 0.125
    step(optimizer, scheduler)
    assert lrs(optimizer) == [0.9375, 0.15625]
    assert scheduler.get_last_lr() == lrs(optimizer)


def test_completed_schedule_keeps_manual_lr_changes(device_scope):
    optimizer, scheduler = make_schedule(device_scope)
    for _ in range(4):
        step(optimizer, scheduler)
    optimizer.param_groups[0]["lr"] = 0.75
    optimizer.param_groups[1]["lr"] = 0.125
    step(optimizer, scheduler)
    assert lrs(optimizer) == [0.75, 0.125]
    assert scheduler.get_last_lr() == [0.75, 0.125]


def test_explicit_epoch_uses_base_lrs_after_manual_change(device_scope):
    optimizer, scheduler = make_schedule(device_scope)
    optimizer.param_groups[0]["lr"] = 0.75
    optimizer.param_groups[1]["lr"] = 0.125
    step(optimizer, scheduler, epoch=2)
    assert scheduler.last_epoch == 2
    assert lrs(optimizer) == [0.15000000000000002, 0.30000000000000004]


def test_restore_keeps_live_lr_for_next_step(device_scope):
    optimizer, scheduler = make_schedule(device_scope)
    optimizer.param_groups[0]["lr"] = 0.75
    optimizer.param_groups[1]["lr"] = 0.125
    saved_optimizer = copy.deepcopy(optimizer.state_dict())
    saved_scheduler = copy.deepcopy(scheduler.state_dict())
    restored_optimizer, restored_scheduler = make_schedule(device_scope)
    restored_optimizer.load_state_dict(saved_optimizer)
    restored_scheduler.load_state_dict(saved_scheduler)
    step(optimizer, scheduler)
    step(restored_optimizer, restored_scheduler)
    assert lrs(restored_optimizer) == lrs(optimizer) == [0.9375, 0.15625]
    assert restored_scheduler.state_dict() == scheduler.state_dict()
    assert restored_scheduler.get_last_lr() == scheduler.get_last_lr()


def test_zero_total_iters_applies_start_factor_then_keeps_live_lr(device_scope):
    optimizer, scheduler = make_schedule(device_scope, total_iters=0)
    assert lrs(optimizer) == [0.1, 0.2]
    step(optimizer, scheduler)
    assert lrs(optimizer) == [0.1, 0.2]


def test_zero_end_factor_does_not_reset_a_later_manual_change(device_scope):
    optimizer, scheduler = make_schedule(device_scope, end_factor=0.0, total_iters=2)
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
def test_factor_range_is_validated_before_initial_step(device_scope, options):
    with pytest.raises(ValueError):
        make_schedule(device_scope, **options)


def test_vision_decay_has_exact_reference_float_trajectory(device_scope):
    # Values from independent torch==2.6.0+cu124; strict scheduler state uses
    # these float values, including the accumulated final rounding.
    expected = {0: 0.01, 1: 0.00991, 2: 0.00982, 3: 0.00973,
                50: 0.005500000000000003, 99: 0.0010900000000000005,
                100: 0.0010000000000000005, 101: 0.0010000000000000005}
    parameter = torch.nn.Parameter(torch.tensor([1.0], device=device_scope))
    assert parameter.device.type == device_scope
    optimizer = torch.optim.SGD([parameter], lr=0.01)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.1, total_iters=100)
    for update in range(102):
        if update in expected:
            assert scheduler.get_last_lr() == [expected[update]], update
            assert lrs(optimizer) == [expected[update]], update
        if update < 101:
            step(optimizer, scheduler)
