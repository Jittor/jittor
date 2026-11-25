"""A checkpoint loader can defer Linear init without losing parameter identity."""

from contextvars import Context

import numpy as np
import pytest
import torch

from jittor.compat.torch.checkpoint_linear import load_with_deferred_linear_initialization


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_checkpoint_assign_materializes_tied_linear_weight(dtype):
    expected = [[1.0, 2.0], [3.0, 4.0]]

    def load():
        model = torch.nn.Module()
        model.first = torch.nn.Linear(2, 2, bias=False, dtype=dtype)
        model.second = torch.nn.Linear(2, 2, bias=False, dtype=dtype)
        original = model.first.weight
        assert original.is_meta and original.dtype is dtype
        model.second.weight = original
        payload = torch.tensor(expected, dtype=dtype)
        with pytest.raises(RuntimeError, match="Cannot assign meta checkpoint storage"):
            model.load_state_dict({"first.weight": payload,
                                   "second.weight": torch.empty(2, 2, device="meta", dtype=dtype)}, assign=True)
        assert original.is_meta
        # An independent loader context also has to clear the holder mark.
        Context().run(model.first.load_state_dict, {"weight": payload}, assign=True)
        assert model.first.weight is original
        return model

    model = load_with_deferred_linear_initialization(load)
    weight = model.first.weight
    assert weight is model.second.weight and weight.device.type == "cpu"
    assert weight.dtype is dtype and weight.requires_grad
    np.testing.assert_array_equal(weight.detach().float().numpy(), expected)


def test_incomplete_checkpoint_reports_unmaterialized_parameter():
    def load():
        layer = torch.nn.Linear(2, 2)
        layer.load_state_dict({"weight": torch.ones(2, 2)}, strict=False, assign=True)
        return layer

    with pytest.raises(RuntimeError, match="left Linear parameters on meta: bias"):
        load_with_deferred_linear_initialization(load)
    ordinary = torch.nn.Linear(2, 2)
    assert not ordinary.weight.is_meta and not ordinary.bias.is_meta


def test_checkpoint_assign_from_module_override():
    class Parent(torch.nn.Module):
        def _load_from_state_dict(self, *args, **kwargs):
            super()._load_from_state_dict(*args, **kwargs)

    def load():
        model = Parent()
        model.linear = torch.nn.Linear(2, 2)
        weight = model.linear.weight
        model.load_state_dict(
            {"linear.weight": torch.ones(2, 2), "linear.bias": torch.zeros(2)},
            assign=True,
        )
        assert model.linear.weight is weight
        return model

    model = load_with_deferred_linear_initialization(load)
    np.testing.assert_array_equal(model.linear.weight.detach().numpy(), [[1, 1], [1, 1]])
