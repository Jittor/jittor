"""Public FULL checkpoint offload materializes every rank before dropping it.

These are host ordering contracts. Real tensors and distributed continuation
are exercised separately by the checkpoint state and hardware gates.
"""

import jittor.compat.torch.installers.distributed as legacy
from jittor.compat.torch import checkpoint_state as owner
import torch


def test_private_and_public_state_spellings_share_the_canonical_owner():
    state = torch.distributed.checkpoint.state_dict
    for name in ("get_model_state_dict", "set_model_state_dict",
                 "get_optimizer_state_dict", "set_optimizer_state_dict",
                 "get_state_dict", "set_state_dict"):
        assert getattr(state, name) is getattr(owner, name)
    for old, name in (("_get_model_state_dict", "get_model_state_dict"),
                      ("_set_model_state_dict", "set_model_state_dict"),
                      ("_get_optimizer_state_dict", "get_optimizer_state_dict"),
                      ("_set_optimizer_state_dict", "set_optimizer_state_dict"),
                      ("_api_checkpoint_sd_get_state_dict", "get_state_dict"),
                      ("_api_checkpoint_sd_set_state_dict", "set_state_dict")):
        assert getattr(legacy, old) is getattr(owner, name)


def offload_probe(monkeypatch, events):
    class Probe:
        def cpu(self):
            events.append("offload")
            return self

        def sync(self):
            events.append("materialize")

    def map_tree(value, function):
        if isinstance(value, dict):
            return {key: map_tree(item, function) for key, item in value.items()}
        if isinstance(value, list):
            return [map_tree(item, function) for item in value]
        return function(value)

    dist = torch.distributed
    monkeypatch.setattr(owner, "_map_tree", map_tree)
    monkeypatch.setattr(owner, "_unwrapped_model", lambda model: model)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)

    def get_rank():
        events.append("drop-check")
        return 1

    monkeypatch.setattr(dist, "get_rank", get_rank)
    return Probe()


def test_public_full_model_offload_materializes_before_nonzero_rank_drop(monkeypatch):
    events = []
    probe = offload_probe(monkeypatch, events)

    class Provider:
        def _get_full_state_dict(self, model):
            events.append("gather")
            return {"weight": probe}

    monkeypatch.setattr(owner, "_fsdp", lambda model: Provider())
    options = legacy.StateDictOptions(full_state_dict=True, cpu_offload=True)
    result = torch.distributed.checkpoint.state_dict.get_model_state_dict(
        object(), options=options)
    assert result == {}
    assert events == ["gather", "offload", "materialize", "drop-check"]


def test_public_full_optimizer_offload_materializes_before_nonzero_rank_drop(monkeypatch):
    events = []
    probe = offload_probe(monkeypatch, events)

    class Optimizer:
        param_groups = []

        def state_dict(self):
            return {"state": {}, "param_groups": []}

    class Provider:
        def refresh_optimizer_fsdp_params(self, optimizer):
            assert isinstance(optimizer, Optimizer)
            events.append("refresh")

        def _gather_optimizer_state_dict(self, model, state, ranks):
            assert ranks == [{}, {}]
            events.append("optimizer-gather")
            return {"weight": probe}

    def gather(ranks, value):
        events.append("metadata-gather")
        ranks[:] = [value, value]

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    monkeypatch.setattr(owner, "_fsdp", lambda model: Provider())
    monkeypatch.setattr(owner, "_parameter_names", lambda model: {})
    monkeypatch.setattr(owner, "_initialize", lambda optimizer: events.append("initialize"))
    options = legacy.StateDictOptions(full_state_dict=True, cpu_offload=True)
    result = torch.distributed.checkpoint.state_dict.get_optimizer_state_dict(
        object(), Optimizer(), options=options)
    assert result == {}
    assert events == ["initialize", "refresh", "metadata-gather", "optimizer-gather",
                      "offload", "materialize", "drop-check"]
