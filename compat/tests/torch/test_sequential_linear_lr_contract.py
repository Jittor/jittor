"""Public LinearLR child behavior in SequentialLR, frozen against Torch 2.6.

These contracts cover live learning rates through the first milestone and
after the final decay. SequentialLR checkpoint interchange is a separate gate.
"""

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


def make_schedule(device):
    parameters = [torch.nn.Parameter(torch.tensor([1.0], device=device)),
                  torch.nn.Parameter(torch.tensor([2.0], device=device))]
    assert all(parameter.device.type == device for parameter in parameters)
    optimizer = torch.optim.SGD([{"params": [parameters[0]], "lr": 0.2},
                                 {"params": [parameters[1]], "lr": 0.4}])
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.5, end_factor=1.0, total_iters=2)
    decay = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.25, total_iters=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, decay], milestones=[2])
    return optimizer, scheduler, warmup, decay


def step(optimizer, scheduler):
    optimizer.step()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        scheduler.step()


def lrs(optimizer):
    return [group["lr"] for group in optimizer.param_groups]


def test_linear_children_have_exact_multigroup_trajectory(device_scope):
    optimizer, scheduler, warmup, decay = make_schedule(device_scope)
    expected = [0.1, 0.15000000000000002, 0.2, 0.1625, 0.125,
                0.0875, 0.049999999999999996, 0.049999999999999996]
    for update, value in enumerate(expected):
        assert scheduler.last_epoch == update
        assert lrs(optimizer) == scheduler.get_last_lr() == [value, value * 2]
        child = warmup if update < 2 else decay
        assert child.get_last_lr() == scheduler.get_last_lr()
        if update + 1 < len(expected):
            step(optimizer, scheduler)


def test_first_linear_child_uses_live_lr_before_milestone(device_scope):
    optimizer, scheduler, warmup, _ = make_schedule(device_scope)
    optimizer.param_groups[0]["lr"] = 0.75
    optimizer.param_groups[1]["lr"] = 0.125
    step(optimizer, scheduler)
    assert lrs(optimizer) == scheduler.get_last_lr() == [1.125, 0.1875]
    assert warmup.get_last_lr() == scheduler.get_last_lr()


def test_milestone_resets_second_linear_child_from_initial_lr(device_scope):
    optimizer, scheduler, _, decay = make_schedule(device_scope)
    step(optimizer, scheduler)
    optimizer.param_groups[0]["lr"] = 0.75
    optimizer.param_groups[1]["lr"] = 0.125
    step(optimizer, scheduler)
    assert scheduler.last_epoch == 2
    assert decay.last_epoch == 0
    assert lrs(optimizer) == scheduler.get_last_lr() == [0.2, 0.4]
    assert decay.get_last_lr() == scheduler.get_last_lr()


def test_completed_second_linear_child_keeps_live_lr(device_scope):
    optimizer, scheduler, _, decay = make_schedule(device_scope)
    for _ in range(6):
        step(optimizer, scheduler)
    optimizer.param_groups[0]["lr"] = 0.75
    optimizer.param_groups[1]["lr"] = 0.125
    step(optimizer, scheduler)
    assert scheduler.last_epoch == 7
    assert decay.last_epoch == 5
    assert lrs(optimizer) == scheduler.get_last_lr() == [0.75, 0.125]
    assert decay.get_last_lr() == scheduler.get_last_lr()
