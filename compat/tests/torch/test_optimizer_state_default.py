"""Optimizer state indexed before the first step, as DeepSpeed ZeRO does."""

import pytest
import torch


def test_registered_parameter_has_empty_state_before_first_step():
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    optimizer = torch.optim.AdamW([parameter], lr=0.01)
    assert optimizer.state.get(parameter) is None
    assert optimizer.state[parameter] == {}
    assert optimizer.state[parameter] == {}

    stranger = torch.nn.Parameter(torch.tensor([3.0]))
    with pytest.raises(KeyError):
        _ = optimizer.state[stranger]

def test_adamw_replaced_parameter_gets_matching_lazy_moments():
    original = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    optimizer = torch.optim.AdamW([original], lr=0.01)
    group = optimizer.param_groups[0]
    assert group["m"] == [None]
    assert group["values"] == [None]

    partition = torch.nn.Parameter(torch.tensor([3.0, 4.0, 5.0]))
    group["params"] = [partition]
    (partition * partition).sum().backward()
    optimizer.step()
    assert list(group["m"][0].shape) == [3]
    assert list(group["values"][0].shape) == [3]
    assert list(optimizer.state[partition]["exp_avg"].shape) == [3]