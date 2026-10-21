"""Known-parameter optimizer state follows persistent default-mapping semantics."""
import pytest
import jittor as jt
from jittor.compat.torch.tensor_state import compatibility_owner


def _make(name, values=(1., -2.)):
    torch = compatibility_owner(jt)
    parameter = torch.tensor(list(values), requires_grad=True)
    optimizer = getattr(torch.optim, name)([parameter], lr=.01)
    return torch, parameter, optimizer


@pytest.mark.parametrize("name", ("Adam", "AdamW"))
def test_getitem_creates_one_persistent_empty_parameter_entry(name):
    _, parameter, optimizer = _make(name)
    state = optimizer.state
    marker = object()
    assert len(state) == 0 and parameter not in state
    assert state.get(parameter, marker) is marker
    assert len(state) == 0
    entry = state[parameter]
    assert entry == {}
    assert optimizer.state is state
    assert optimizer.state[parameter] is entry
    assert state.get(parameter) is entry
    assert len(state) == 1 and parameter in state
    assert list(state)[0] is parameter
    assert list(state.keys())[0] is parameter
    assert list(state.values())[0] is entry
    assert list(state.items())[0][0] is parameter and list(state.items())[0][1] is entry
    saved = optimizer.state_dict()
    assert saved["state"] == {0: {}}
    assert saved["param_groups"][0]["decoupled_weight_decay"] is (name == "AdamW")


@pytest.mark.parametrize("name", ("Adam", "AdamW"))
def test_empty_state_save_load_preserves_presence_without_inventing_other_entries(name):
    torch, parameter, optimizer = _make(name)
    other = torch.tensor([3.], requires_grad=True)
    optimizer.add_param_group({"params": [other]})
    optimizer.state[parameter]
    saved = optimizer.state_dict()
    assert saved["state"] == {0: {}}
    _, restored_parameter, restored = _make(name)
    restored_other = torch.tensor([3.], requires_grad=True)
    restored.add_param_group({"params": [restored_other]})
    old = restored.state[restored_other]
    restored.load_state_dict(saved)
    assert len(restored.state) == 1
    assert restored_parameter in restored.state and restored_other not in restored.state
    assert restored.state[restored_parameter] == {}
    assert old == {}
    assert restored.state_dict()["state"] == {0: {}}
    assert restored.state_dict()["param_groups"][1]["decoupled_weight_decay"] is (name == "AdamW")


@pytest.mark.parametrize("name", ("Adam", "AdamW"))
def test_retained_empty_dictionary_observes_updates_and_nested_writes(name):
    torch, parameter, optimizer = _make(name)
    entry = optimizer.state[parameter]
    (parameter * parameter).sum().backward()
    optimizer.step()
    assert entry is optimizer.state[parameter]
    assert set(entry) == {"exp_avg", "exp_avg_sq", "step"}
    assert float(entry["step"]) == 1
    replacement = torch.zeros_like(entry["exp_avg"])
    entry["exp_avg"] = replacement
    entry["annotation"] = {"source": "public-state"}
    assert optimizer.state[parameter]["exp_avg"] is replacement
    optimizer.zero_grad(set_to_none=True)
    (parameter * parameter).sum().backward()
    optimizer.step()
    assert float(entry.get("step")) == 2
    assert entry is optimizer.state[parameter]
    assert dict(entry)["annotation"] == {"source": "public-state"}
    assert optimizer.state_dict()["state"][0]["annotation"] == {"source": "public-state"}
    _, restored_parameter, restored = _make(name)
    restored.load_state_dict(optimizer.state_dict())
    assert restored.state[restored_parameter]["annotation"] == {"source": "public-state"}
    assert float(restored.state[restored_parameter]["step"]) == 2


@pytest.mark.parametrize("name", ("Adam", "AdamW"))
def test_empty_assignment_delete_clear_and_late_groups_share_presence(name):
    torch, parameter, optimizer = _make(name)
    optimizer.state[parameter] = {}
    assert len(optimizer.state) == 1 and parameter in optimizer.state
    original = optimizer.state[parameter]
    del optimizer.state[parameter]
    assert len(optimizer.state) == 0 and parameter not in optimizer.state
    assert optimizer.state_dict()["state"] == {}
    assert optimizer.state[parameter] is not original
    optimizer.state.clear()
    assert len(optimizer.state) == 0 and optimizer.state_dict()["state"] == {}
    late = torch.tensor([3.], requires_grad=True)
    optimizer.add_param_group({"params": [late]})
    optimizer.state[late]
    assert late in optimizer.state and parameter not in optimizer.state
    assert len(optimizer.state) == 1
    assert optimizer.state_dict()["state"] == {1: {}}


@pytest.mark.parametrize("name", ("Adam", "AdamW"))
def test_foreign_parameter_does_not_create_a_known_parameter_state(name):
    torch, _, optimizer = _make(name)
    foreign = torch.tensor([7.], requires_grad=True)
    marker = object()
    assert optimizer.state.get(foreign, marker) is marker
    assert foreign not in optimizer.state
    with pytest.raises(KeyError):
        optimizer.state[foreign]
    with pytest.raises(KeyError):
        optimizer.state[foreign] = {}
    assert len(optimizer.state) == 0 and optimizer.state_dict()["state"] == {}
