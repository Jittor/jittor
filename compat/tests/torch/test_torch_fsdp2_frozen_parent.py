"""A frozen FSDP parent must preserve trainable child gradients."""

from types import SimpleNamespace
from unittest import mock

import jittor as jt
import pytest
import torch

from jittor.compat.fsdp2 import shard


@pytest.fixture(autouse=True)
def cpu_only():
    with jt.flag_scope(use_cuda=0):
        yield


def test_frozen_parent_keeps_trainable_child_output_connected():
    class Child(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([2.]))

        def forward(self, value):
            return value * self.weight

    class Parent(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.frozen = torch.nn.Parameter(torch.tensor([1.]), requires_grad=False)
            self.child = Child()

        def forward(self, value):
            return self.child(value) + self.frozen

    parent = Parent()
    parent._fsdp_state = SimpleNamespace(
        true_fsdp_initialized=True,
        true_fsdp_params=(SimpleNamespace(requires_grad=False),),
        reshard_after_forward=True,
    )
    input_value = torch.tensor([3.], requires_grad=False)
    with mock.patch.object(shard, "_unshard_module_params"), \
            mock.patch.object(shard, "_reshard_module_params"), \
            mock.patch.object(shard.jt, "gc"):
        result = shard._execute_with_true_fsdp(parent, parent.forward, input_value)
    (gradient,) = jt.core.grad_optional(result.sum(), [parent.child.weight], False)
    assert gradient is not None
    assert float(gradient.item()) == pytest.approx(3.)
