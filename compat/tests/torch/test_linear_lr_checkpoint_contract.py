"""LinearLR public checkpoint and warning contracts from PyTorch 2.6."""

import copy
import warnings

import pytest
import torch

from test_linear_lr_public_contract import device_scope, lrs, make_schedule, step


def test_default_public_state_has_reference_checkpoint_schema(device_scope):
    _, scheduler = make_schedule(device_scope)
    assert scheduler.state_dict() == {
        "start_factor": 0.5, "end_factor": 1.0, "total_iters": 4,
        "base_lrs": [0.2, 0.4], "last_epoch": 0, "verbose": False,
        "_step_count": 1, "_get_lr_called_within_step": False,
        "_last_lr": [0.1, 0.2],
    }


@pytest.mark.parametrize("verbose", [False, True])
def test_explicit_verbose_warns_and_is_saved(device_scope, verbose):
    optimizer, _ = make_schedule(device_scope)
    with pytest.warns(UserWarning, match="verbose parameter is deprecated"):
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, verbose=verbose)
    assert scheduler.state_dict()["verbose"] is verbose


def test_default_verbose_and_normal_step_do_not_warn(device_scope):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        optimizer, scheduler = make_schedule(device_scope)
        step(optimizer, scheduler)
    assert not caught, [str(warning.message) for warning in caught]


def test_get_lr_warns_outside_step(device_scope):
    optimizer, scheduler = make_schedule(device_scope)
    with pytest.warns(UserWarning, match="get_last_lr"):
        scheduler.get_lr()
    assert scheduler.get_last_lr() == lrs(optimizer) == [0.1, 0.2]


def test_get_lr_observes_step_context_then_resets_it(device_scope):
    optimizer, _ = make_schedule(device_scope)

    class Observed(torch.optim.lr_scheduler.LinearLR):
        def __init__(self, *args, **kwargs):
            self.observed = []
            super().__init__(*args, **kwargs)

        def get_lr(self):
            self.observed.append(self.state_dict().get("_get_lr_called_within_step"))
            return super().get_lr()

    scheduler = Observed(optimizer)
    assert scheduler.observed == [True]
    assert scheduler.state_dict()["_get_lr_called_within_step"] is False
    step(optimizer, scheduler)
    assert scheduler.observed == [True, True]
    assert scheduler.state_dict()["_get_lr_called_within_step"] is False


def test_step_context_resets_after_get_lr_raises(device_scope):
    optimizer, _ = make_schedule(device_scope)

    class Failing(torch.optim.lr_scheduler.LinearLR):
        def get_lr(self):
            if self.last_epoch > 0:
                assert self.state_dict()["_get_lr_called_within_step"] is True
                raise RuntimeError("scheduled failure")
            return super().get_lr()

    scheduler = Failing(optimizer)
    with pytest.raises(RuntimeError, match="scheduled failure"):
        step(optimizer, scheduler)
    assert scheduler.state_dict()["_get_lr_called_within_step"] is False


def test_initial_lr_is_preserved_for_explicit_epoch(device_scope):
    optimizer, _ = make_schedule(device_scope)
    assert [group["initial_lr"] for group in optimizer.param_groups] == [0.2, 0.4]
    optimizer.param_groups[0]["lr"] = 0.75
    optimizer.param_groups[1]["lr"] = 0.125
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.5, end_factor=1.0, total_iters=4)
    assert scheduler.base_lrs == [0.2, 0.4]
    assert lrs(optimizer) == [0.375, 0.0625]
    step(optimizer, scheduler, epoch=2)
    assert lrs(optimizer) == [0.15000000000000002, 0.30000000000000004]


def test_resumed_constructor_requires_initial_lr_in_all_groups(device_scope):
    optimizer, _ = make_schedule(device_scope)
    del optimizer.param_groups[1]["initial_lr"]
    with pytest.raises(KeyError, match=r"initial_lr.*param_groups\[1\]"):
        torch.optim.lr_scheduler.LinearLR(optimizer, last_epoch=2)


def test_resumed_constructor_uses_current_lr_and_saved_base_lrs(device_scope):
    optimizer, _ = make_schedule(device_scope)
    optimizer.param_groups[0]["lr"] = 0.75
    optimizer.param_groups[1]["lr"] = 0.125
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.5, end_factor=1.0, total_iters=4, last_epoch=2)
    assert scheduler.last_epoch == 3
    assert scheduler.base_lrs == [0.2, 0.4]
    assert lrs(optimizer) == [0.875, 0.14583333333333334]
    step(optimizer, scheduler, epoch=2)
    assert lrs(optimizer) == [0.15000000000000002, 0.30000000000000004]


def test_weights_only_checkpoint_keeps_reference_metadata_and_next_lr(device_scope, tmp_path):
    optimizer, scheduler = make_schedule(device_scope)
    optimizer.param_groups[0]["lr"] = 0.75
    optimizer.param_groups[1]["lr"] = 0.125
    checkpoint = {"optimizer": copy.deepcopy(optimizer.state_dict()),
                  "scheduler": copy.deepcopy(scheduler.state_dict())}
    target = tmp_path / "linear-lr.pt"
    torch.save(checkpoint, target)
    loaded = torch.load(target, weights_only=True)
    assert loaded["scheduler"]["verbose"] is False
    assert loaded["scheduler"]["_get_lr_called_within_step"] is False
    assert [group["initial_lr"] for group in loaded["optimizer"]["param_groups"]] == [0.2, 0.4]
    restored_optimizer, restored_scheduler = make_schedule(device_scope)
    restored_optimizer.load_state_dict(loaded["optimizer"])
    restored_scheduler.load_state_dict(loaded["scheduler"])
    assert restored_scheduler.state_dict() == scheduler.state_dict()
    step(optimizer, scheduler)
    step(restored_optimizer, restored_scheduler)
    assert lrs(optimizer) == lrs(restored_optimizer) == [0.9375, 0.15625]
    assert restored_scheduler.state_dict() == scheduler.state_dict()
