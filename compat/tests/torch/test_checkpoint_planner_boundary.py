"""FULL checkpoint imports cannot silently enable sharded checkpoint planning."""

import importlib

import pytest
import torch

from jittor.compat.torch.fidelity import Fidelity, fidelity_of


@pytest.mark.parametrize("name", ["DefaultSavePlanner", "DefaultLoadPlanner"])
def test_default_planner_import_is_explicitly_unsupported(name, monkeypatch):
    monkeypatch.setenv("JITTOR_TORCH_ALLOW_STUB", "1")
    module = importlib.import_module("torch.distributed.checkpoint.default_planner")
    assert torch.distributed.checkpoint.default_planner is module
    planner = getattr(module, name)
    record = fidelity_of("torch.distributed.checkpoint.default_planner." + name)
    assert record.level is Fidelity.UNIMPLEMENTED
    assert record.implementation is planner
    with pytest.raises(NotImplementedError, match="sharded checkpoint planning"):
        planner()
