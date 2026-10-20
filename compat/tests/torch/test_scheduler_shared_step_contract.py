"""Shared step metadata belongs to the canonical scheduler base."""

import torch
import pytest

from test_linear_lr_public_contract import device_scope, make_schedule, step


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
def test_shared_step_context_is_active_only_during_get_lr(device_scope, name, options):
    optimizer, _ = make_schedule(device_scope)
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
    assert scheduler.state_dict()["verbose"] is False
    step(optimizer, scheduler)
    assert scheduler.observed == [True, True]
    assert scheduler.state_dict()["_get_lr_called_within_step"] is False


def test_shared_step_resets_context_after_exception(device_scope):
    optimizer, _ = make_schedule(device_scope)

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


def test_loading_legacy_scheduler_state_keeps_new_constructor_defaults(device_scope):
    optimizer, _ = make_schedule(device_scope)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.5 ** epoch)
    legacy = scheduler.state_dict()
    legacy.pop("verbose")
    legacy.pop("_get_lr_called_within_step")
    restored = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.5 ** epoch)
    restored.load_state_dict(legacy)
    assert restored.state_dict()["verbose"] is False
    assert restored.state_dict()["_get_lr_called_within_step"] is False
    # Loading remains allowed; a strict external old/new snapshot comparison
    # must still report the schema change and establish a fresh run baseline.
    assert set(restored.state_dict()) == set(legacy) | {"verbose", "_get_lr_called_within_step"}
