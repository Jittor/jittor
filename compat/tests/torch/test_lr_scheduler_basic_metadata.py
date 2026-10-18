"""Shared-base scheduler regressions with float metadata, no tensors or models."""
import copy
import pytest
import torch


class MetadataOptimizer(torch.optim.Optimizer):
    # This fixture deliberately bypasses native optimizer construction.
    defaults = None
    def __init__(self, groups, defaults=None):
        self.param_groups = groups
        self.defaults = defaults or {}
        self.lr = groups[0]["lr"]
    def step(self):
        pass


def test_step_lr_basic_trajectory_and_checkpoint_continuation():
    optimizer = MetadataOptimizer([{"lr": 0.125}])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    assert scheduler.base_lrs == [0.125]
    assert optimizer.param_groups[0]["initial_lr"] == 0.125
    rates = [scheduler.get_last_lr()[0]]
    for _ in range(2):
        optimizer.step(); scheduler.step()
        rates.append(scheduler.get_last_lr()[0])
    saved = copy.deepcopy(scheduler.state_dict())
    restored_optimizer = MetadataOptimizer([{"lr": 0.125}])
    restored = torch.optim.lr_scheduler.StepLR(restored_optimizer, step_size=2, gamma=0.5)
    restored_optimizer.param_groups = copy.deepcopy(optimizer.param_groups)
    restored.load_state_dict(saved)
    for _ in range(2):
        optimizer.step(); scheduler.step()
        restored_optimizer.step(); restored.step()
        rates.append(scheduler.get_last_lr()[0])
        assert restored.state_dict() == scheduler.state_dict()
        assert restored_optimizer.param_groups == optimizer.param_groups
    assert rates == [0.125, 0.125, 0.0625, 0.0625, 0.03125]


@pytest.mark.parametrize("mode", ["momentum", "betas"])
def test_one_cycle_precomputed_base_and_float_trajectory(mode):
    settings = {"momentum": 0.75} if mode == "momentum" else {"betas": (0.75, 0.99)}
    optimizer = MetadataOptimizer([{"lr": 0.9, "initial_lr": 9.0, **settings}], settings)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1.25, total_steps=4, pct_start=0.5,
        div_factor=10.0, final_div_factor=4.0, anneal_strategy="linear",
        base_momentum=0.5, max_momentum=0.875)
    assert scheduler.base_lrs == [0.125]
    group = optimizer.param_groups[0]
    assert (group["initial_lr"], group["max_lr"], group["min_lr"]) == (0.125, 1.25, 0.03125)
    rates, momentums = [], []
    for index in range(4):
        if index:
            optimizer.step(); scheduler.step()
        rates.append(scheduler.get_last_lr()[0])
        momentums.append(group["momentum"] if mode == "momentum" else group["betas"][0])
        if mode == "betas":
            assert group["betas"][1] == 0.99
        assert scheduler._is_initial is False
        assert scheduler._get_lr_called_within_step is False
    assert rates == [0.125, 1.25, 0.640625, 0.03125]
    assert momentums == [0.875, 0.5, 0.6875, 0.875]
