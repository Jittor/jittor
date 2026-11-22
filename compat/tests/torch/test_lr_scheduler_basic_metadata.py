"""Scheduler metadata, phase boundaries and real optimizer update regressions."""
import copy
import os
import numpy as np
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


# Native Torch 2.11 CPU trajectories, including the final scheduler call after
# the eighth optimizer update. Linear schedules extrapolate below zero there.
@pytest.mark.parametrize("three_phase,strategy,rates,momenta", [
    (False, "linear", [.25, .5, .75, 1., .765625, .53125, .296875, .0625, -.171875],
     [.875, .75, .625, .5, .59375, .6875, .78125, .875, .96875]),
    (True, "linear", [.25, .5, .75, 1., .75, .5, .25, .0625, -.125],
     [.875, .75, .625, .5, .625, .75, .875, .875, .875]),
    (False, "cos", [.25, .4375, .8125, 1., .8627063036811942, .53125,
                    .1997936963188059, .0625, .19979369631880578],
     [.875, .78125, .59375, .5, .5549174785275224, .6875, .8200825214724776,
      .875, .8200825214724777]),
    (True, "cos", [.25, .4375, .8125, 1., .8125, .4375, .25, .0625, .25],
     [.875, .78125, .59375, .5, .59375, .78125, .875, .875, .875]),
])
@pytest.mark.parametrize("mode", ["momentum", "betas", "off"])
def test_one_cycle_terminal_values_overrun_and_checkpoint(three_phase, strategy, rates, momenta, mode):
    defaults = {"betas": (.75, .99)} if mode == "betas" else {"momentum": .75}
    optimizer = MetadataOptimizer([{"lr": .1, **defaults}, {"lr": .2, **defaults}], defaults)
    options = dict(max_lr=[1., .5], total_steps=8, pct_start=.5, div_factor=4.,
                   final_div_factor=4., base_momentum=.5, max_momentum=.875,
                   three_phase=three_phase, anneal_strategy=strategy, cycle_momentum=mode != "off")
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, **options)
    for index in range(9):
        if index:
            optimizer.step()
            scheduler.step()
        assert scheduler.last_epoch == index
        assert scheduler.get_last_lr() == pytest.approx([rates[index], rates[index]/2], abs=1e-14)
        for group in optimizer.param_groups:
            if mode == "betas":
                assert group["betas"] == pytest.approx((momenta[index], .99), abs=1e-14)
            else:
                assert group["momentum"] == pytest.approx(.75 if mode == "off" else momenta[index], abs=1e-14)
        if index == 7:
            saved = copy.deepcopy(scheduler.state_dict())
            groups = copy.deepcopy(optimizer.param_groups)
    restored_optimizer = MetadataOptimizer(copy.deepcopy(groups), defaults)
    restored = torch.optim.lr_scheduler.OneCycleLR(restored_optimizer, **options)
    restored_optimizer.param_groups = groups
    restored.load_state_dict(saved)
    restored_optimizer.step()
    restored.step()
    assert restored.get_last_lr() == scheduler.get_last_lr()
    assert restored_optimizer.param_groups == optimizer.param_groups
    for value in (scheduler, restored):
        with pytest.raises(ValueError, match=r"Tried to step 9 times.*total steps is 8"):
            value.step()
        assert value.last_epoch == 9
        assert value.get_last_lr() == pytest.approx([rates[-1], rates[-1]/2], abs=1e-14)


@pytest.mark.parametrize("device", os.environ.get("JITTOR_TEST_DEVICES", "cpu").split(","))
def test_one_cycle_uses_valid_rates_for_real_optimizer_updates(device):
    import jittor as jt
    before = jt.core.backend_fallback_count()
    with jt.flag_scope(backend_fallback="error"):
        parameter = torch.nn.Parameter(torch.ones(2, device=device))
        optimizer = torch.optim.SGD([parameter], lr=.1)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1., total_steps=8,
            pct_start=.5, div_factor=4., final_div_factor=4., cycle_momentum=False, anneal_strategy="linear")
        rates = [.25, .5, .75, 1., .765625, .53125, .296875, .0625]
        gradient = np.array([.25, -.5], dtype="float32")
        expected = np.ones(2, dtype="float32")
        for rate in rates:
            assert optimizer.param_groups[0]["lr"] == rate
            parameter.grad = torch.tensor(gradient, device=device)
            optimizer.step()
            scheduler.step()
            expected -= np.float32(rate) * gradient
            parameter.sync()
            assert parameter.device.type == device
            assert parameter.location() == ("device" if device == "cuda" else "cpu")
            np.testing.assert_allclose(parameter.detach().cpu().numpy(), expected, rtol=1e-6, atol=1e-6)
        assert scheduler.get_last_lr() == [-.171875]
        with pytest.raises(ValueError, match="Tried to step 9 times"):
            scheduler.step()
    assert jt.core.backend_fallback_count() == before
