"""Versioned child modules migrate legacy checkpoint keys before loading."""

from collections import OrderedDict

import pytest
import torch


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    return request.param


class Versioned(torch.nn.Module):
    _version = 2

    def __init__(self, device):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(3, device=device, dtype=torch.float64))
        self.pre_hooks = []
        self.register_load_state_dict_pre_hook(self.record_load)

    def record_load(self, module, state, prefix, metadata, strict, missing, unexpected, errors):
        self.pre_hooks.append((prefix, tuple(state)))

    def _load_from_state_dict(self, state, prefix, metadata, strict, missing, unexpected, errors):
        self.seen = (prefix, metadata.get("version"))
        old = prefix + "legacy"
        if metadata.get("version", 1) < 2 and old in state:
            state[prefix + "weight"] = state.pop(old)
        super()._load_from_state_dict(state, prefix, metadata, strict, missing, unexpected, errors)
        self.loaded = self.weight.detach().clone()


@pytest.mark.parametrize("has_metadata", [False, True])
def test_nested_version_migration_preserves_storage_and_input(device, has_metadata):
    child = Versioned(device)
    model = torch.nn.Sequential(child)
    original = child.weight
    alias = original.detach()
    state = OrderedDict([("0.legacy", torch.tensor([1., 2., 3.], device=device))])
    if has_metadata:
        state._metadata = {"0": {"version": 1}}
    result = model.load_state_dict(state)
    assert not result.missing_keys and not result.unexpected_keys
    assert child.seen == ("0.", 1 if has_metadata else None)
    assert child.pre_hooks == [("0.", ("0.weight",))]
    assert tuple(state) == ("0.legacy",)
    assert child.weight is original
    assert alias.data_ptr() == child.weight.data_ptr()
    assert child.weight.requires_grad
    assert child.weight.device.type == device and child.weight.dtype == torch.float64
    expect = torch.tensor([1., 2., 3.], device=device, dtype=torch.float64)
    torch.testing.assert_close(child.loaded, expect)
    torch.testing.assert_close(alias, expect)


def test_nested_migration_keeps_strict_and_shape_errors(device):
    model = torch.nn.Sequential(Versioned(device), Versioned(device))
    state = {"0.legacy": torch.ones(3, device=device), "bogus": torch.ones(1, device=device)}
    result = model.load_state_dict(state, strict=False)
    assert result.missing_keys == ["1.weight"]
    assert result.unexpected_keys == ["bogus"]
    with pytest.raises(RuntimeError, match="Missing key.*1.weight"):
        model.load_state_dict(state)
    with pytest.raises(RuntimeError, match="size mismatch for 0.weight"):
        model.load_state_dict({"0.legacy": torch.ones(2, device=device)}, strict=False)


def test_assign_after_migration_keeps_checkpoint_dtype(device):
    child = Versioned(device)
    state = {"legacy": torch.ones(3, device=device, dtype=torch.float32)}
    result = child.load_state_dict(state, assign=True)
    assert not result.missing_keys and not result.unexpected_keys
    assert child.weight.dtype == torch.float32 and child.weight.requires_grad
    assert child.weight.device.type == device
    torch.testing.assert_close(child.loaded, state["legacy"])


def test_override_can_load_custom_values_without_calling_the_base(device):
    class Custom(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1, device=device))

        def _load_from_state_dict(self, state, prefix, metadata, strict, missing, unexpected, errors):
            with torch.no_grad():
                self.weight.copy_(state[prefix + "custom"] * 2.)

    model = torch.nn.Sequential(Custom(), Versioned(device))
    result = model.load_state_dict({
        "0.custom": torch.tensor([4.], device=device),
        "1.legacy": torch.ones(3, device=device),
    })
    assert not result.missing_keys and not result.unexpected_keys
    assert model[0].weight.item() == 8.
    assert model[1].weight.sum().item() == 3.
