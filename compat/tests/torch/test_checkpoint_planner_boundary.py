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


def test_default_planners_keep_target_positional_constructor_contract():
    module = importlib.import_module("torch.distributed.checkpoint.default_planner")
    save = module.DefaultSavePlanner(True, True, True, None)
    load = module.DefaultLoadPlanner(True, True, False, True)
    assert save.flatten_state_dict and save.flatten_sharded_tensors
    assert save.dedup_replicated_tensors and save.storage_meta is None
    assert load.flatten_state_dict and not load.allow_partial_load


def test_standard_torch_checkpoint_format_is_explicitly_refused(tmp_path):
    dcp = importlib.import_module("torch.distributed.checkpoint")
    (tmp_path / ".metadata").write_bytes(b"foreign-torch-checkpoint")
    with pytest.raises(NotImplementedError, match=r"PyTorch \.metadata"):
        dcp.FileSystemReader(tmp_path).read_metadata()
    with pytest.raises((NotImplementedError, RuntimeError), match=r"PyTorch \.metadata"):
        dcp.load({"weight": torch.zeros(2)}, checkpoint_id=tmp_path)
