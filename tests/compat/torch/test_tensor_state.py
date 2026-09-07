from types import SimpleNamespace
import gc
import weakref

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


def test_tensor_state_does_not_duplicate_native_requires_grad_state():
    module = SimpleNamespace()
    state = get_tensor_state(module)
    assert not hasattr(state, "requires_grad")
    tensor = object()
    state.leaf_params[id(tensor)] = tensor
    state.retained[id(tensor)] = tensor
    assert get_tensor_state(module) is state
    assert module._torch_leaf_params[id(tensor)] is tensor
    assert module._torch_retained[id(tensor)] is tensor


def test_tensor_state_releases_values_removed_from_backward_registries():
    module = SimpleNamespace()
    state = get_tensor_state(module)
    class Tensor:
        pass
    tensor = Tensor()
    key = id(tensor)
    reference = weakref.ref(tensor)
    state.leaf_params[key] = tensor
    state.retained[key] = tensor
    del tensor
    state.leaf_params.pop(key)
    gc.collect()
    assert reference() is state.retained[key]
    state.retained.pop(key)
    gc.collect()
    assert reference() is None
