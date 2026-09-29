"""FSDP parameter discovery follows the module's parameter classification."""

import torch
import pytest
from unittest import mock

from jittor.compat.fsdp2 import shard
from jittor.compat.fsdp2.shard import _named_parameters_with_owner


def test_registered_buffer_alias_does_not_become_trainable_fsdp_entry():
    class Rotary(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(2))
            self.register_buffer("inv_freq", torch.ones(64), persistent=False)
            self.original_inv_freq = self.inv_freq

    module = Rotary()
    # Model loading may replace buffer storage without carrying its Var tags.
    module.inv_freq = torch.ones(64)
    module.original_inv_freq = module.inv_freq
    assert module.original_inv_freq is module.inv_freq
    assert [name for name, _ in module.named_parameters()] == ["weight"]
    assert [name for name, _, _, _ in _named_parameters_with_owner(module)] == ["weight"]


@pytest.mark.parametrize("kind", ["list", "dict"])
def test_parameter_containers_refuse_sharding_before_replacing_values(kind):
    module = torch.nn.Module()
    if kind == "list":
        module.params = torch.nn.ParameterList([torch.nn.Parameter(torch.ones(2))])
    else:
        module.named = torch.nn.ParameterDict({"scale": torch.nn.Parameter(torch.ones(3))})
    entries = _named_parameters_with_owner(module)
    assert [name for name, _, _, _ in entries] == [
        name for name, _ in module.named_parameters()]
    container = module.params if kind == "list" else module.named
    original = container[0] if kind == "list" else container["scale"]
    state = shard.common.StateRecord(shard_group=None, ignored_params=())
    with mock.patch.object(shard.common, "_in_true_distributed", return_value=True), \
            mock.patch.object(shard.common, "_world_size", return_value=2), \
            mock.patch.object(shard.common, "_rank", return_value=0):
        with pytest.raises(NotImplementedError, match="container-aware replacement"):
            shard._init_true_fsdp_state(module, state)
    assert not getattr(state, "true_fsdp_initialized", False)
    assert (container[0] if kind == "list" else container["scale"]) is original
