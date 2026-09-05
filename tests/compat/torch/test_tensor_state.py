from types import SimpleNamespace

from jittor.compat.torch.tensor_state import TorchTensorState, get_tensor_state


def test_tensor_state_owns_optimizer_registry_and_keeps_alias_identity():
    legacy = []
    module = SimpleNamespace(_torch_leaf_params={}, _active_optimizers=legacy)

    state = get_tensor_state(module)

    assert isinstance(state, TorchTensorState)
    assert module._torch_tensor_state is state
    assert state.active_optimizers is legacy
    assert module._active_optimizers is state.active_optimizers
    state.active_optimizers.append("optimizer")
    assert module._active_optimizers == ["optimizer"]


def test_tensor_state_preserves_legacy_leaf_and_retained_aliases():
    module = SimpleNamespace(_torch_leaf_params={"leaf": object()})

    state = get_tensor_state(module)

    assert state.leaf_params is state
    assert "leaf" in state
    assert module._torch_tensor_state is state
    assert module._torch_retained is state.retained


def test_tensor_state_migrates_existing_retained_entries():
    retained = {"retained": object()}
    module = SimpleNamespace(_torch_leaf_params={}, _torch_retained=retained)

    state = get_tensor_state(module)

    assert state.retained == retained
    assert state.retained is not retained
    assert module._torch_retained is state.retained


def test_tensor_state_reuses_explicit_owner_over_legacy_alias():
    state = TorchTensorState()
    legacy = {"stale": object()}
    module = SimpleNamespace(
        _torch_tensor_state=state,
        _torch_leaf_params=legacy,
    )

    assert get_tensor_state(module) is state
    assert module._torch_leaf_params is state
    assert "stale" not in state


def test_tensor_state_owns_requires_grad_lifetime():
    module = SimpleNamespace()
    state = get_tensor_state(module)
    tensor = object()

    assert state.set_requires_grad(tensor, True) is True
    assert state.requires_grad_tensors() == (tensor,)

    assert state.set_requires_grad(tensor, False) is False
    assert state.requires_grad_tensors() == ()


def test_tensor_state_requires_grad_snapshot_isolated_from_registry():
    module = SimpleNamespace()
    state = get_tensor_state(module)
    first = object()
    second = object()
    state.set_requires_grad(first, True)
    state.set_requires_grad(second, True)

    snapshot = state.requires_grad_tensors()
    state.set_requires_grad(first, False)
    assert snapshot == (first, second)
    assert state.requires_grad_tensors() == (second,)
