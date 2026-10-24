"""Default DCP planning has one native owner; custom options remain explicit."""

import importlib

import pytest
import torch

from jittor.compat.torch.fidelity import Fidelity, fidelity_of


@pytest.mark.parametrize("name", ["DefaultSavePlanner", "DefaultLoadPlanner"])
def test_default_planner_is_owned_and_custom_planning_rejects(name, monkeypatch):
    monkeypatch.setenv("JITTOR_TORCH_ALLOW_STUB", "1")
    module = importlib.import_module("torch.distributed.checkpoint.default_planner")
    assert torch.distributed.checkpoint.default_planner is module
    planner = getattr(module, name)
    record = fidelity_of("torch.distributed.checkpoint.default_planner." + name)
    assert record.level is Fidelity.APPROXIMATE
    assert record.implementation is planner
    assert type(planner()) is planner
    with pytest.raises(NotImplementedError, match="default native DCP"):
        planner(flatten_state_dict=False)
