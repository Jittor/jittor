"""Shared step metadata belongs to the canonical scheduler base."""

import torch
import pytest

from _release_test_helpers import (
    linear_schedule as make_schedule,
    public_device_scope, step_scheduler as step,
)


@pytest.fixture
def shared_lr_device():
    with public_device_scope() as name:
        yield name


@pytest.mark.parametrize("name,options", [
    ("LambdaLR", {"lr_lambda": lambda epoch: 0.5 ** epoch}),
    ("MultiplicativeLR", {"lr_lambda": lambda epoch: 0.5}),
    ("ConstantLR", {"factor": 0.5, "total_iters": 3}),
    ("StepLR", {"step_size": 2, "gamma": 0.5}),
    ("MultiStepLR", {"milestones": [2, 4], "gamma": 0.5}),
    ("ExponentialLR", {"gamma": 0.5}),
    ("CosineAnnealingLR", {"T_max": 4}),
    ("PolynomialLR", {"total_iters": 4}),
    ("OneCycleLR", {"max_lr": 0.4, "total_steps": 8, "cycle_momentum": False}),
])
def test_shared_step_context_is_active_only_during_get_lr(shared_lr_device, name, options):
    optimizer, _ = make_schedule(shared_lr_device)
    scheduler_class = getattr(torch.optim.lr_scheduler, name)

    class Observed(scheduler_class):
        def __init__(self, *args, **kwargs):
            self.observed = []
            super().__init__(*args, **kwargs)

        def get_lr(self):
            self.observed.append(self.state_dict().get("_get_lr_called_within_step"))
            return super().get_lr()

    scheduler = Observed(optimizer, **options)
    assert scheduler.observed == [True]
    assert scheduler.state_dict()["_get_lr_called_within_step"] is False
    # Torch 2.7.1 scheduler checkpoints do not serialize verbose for any of
    # these scheduler classes.
    assert "verbose" not in scheduler.state_dict()
    step(optimizer, scheduler)
    assert scheduler.observed == [True, True]
    assert scheduler.state_dict()["_get_lr_called_within_step"] is False


def test_shared_step_resets_context_after_exception(shared_lr_device):
    optimizer, _ = make_schedule(shared_lr_device)

    class Failing(torch.optim.lr_scheduler.StepLR):
        def get_lr(self):
            if self.last_epoch > 0:
                assert self.state_dict()["_get_lr_called_within_step"] is True
                raise RuntimeError("shared scheduled failure")
            return super().get_lr()

    scheduler = Failing(optimizer, step_size=2)
    with pytest.raises(RuntimeError, match="shared scheduled failure"):
        step(optimizer, scheduler)
    assert scheduler.state_dict()["_get_lr_called_within_step"] is False


def test_loading_legacy_linear_scheduler_state_keeps_new_constructor_defaults(shared_lr_device):
    optimizer, _ = make_schedule(shared_lr_device)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.5, total_iters=3)
    legacy = scheduler.state_dict()
    legacy.pop("verbose", None)
    legacy.pop("_get_lr_called_within_step")
    restored = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.5, total_iters=3)
    restored.load_state_dict(legacy)
    assert "verbose" not in restored.state_dict()
    assert restored.state_dict()["_get_lr_called_within_step"] is False
    # Loading remains allowed; a strict external old/new snapshot comparison
    # must still report the schema change and establish a fresh run baseline.
    assert set(restored.state_dict()) == set(legacy) | {"_get_lr_called_within_step"}
