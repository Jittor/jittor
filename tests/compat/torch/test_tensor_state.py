from types import SimpleNamespace
from types import ModuleType
import gc
import weakref
import pytest

from jittor.compat.torch.tensor_state import (
    TorchTensorState, get_tensor_state, compatibility_owner, bind_tensor_state,
    snapshot_tensor_state, record_tensor_state_changes,
)
from jittor.compat.torch.namespace import TorchNamespace
from jittor.compat.transaction import InstallTransaction, TransactionConflict


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


def test_tensor_state_binding_adopts_one_owner_and_rolls_back_aliases():
    native = ModuleType("native_tensor_owner")
    state = get_tensor_state(native)
    state["leaf"] = object()
    state.retained["retained"] = object()
    state.active_optimizers.append(object())
    target = TorchNamespace(native)
    transaction = InstallTransaction("tensor-owner")
    assert bind_tensor_state(native, target, transaction) is state
    assert compatibility_owner(native) is target
    assert compatibility_owner(target) is target
    assert get_tensor_state(native) is get_tensor_state(target) is state
    for owner in (native, target):
        assert vars(owner)["_torch_tensor_state"] is state
        assert vars(owner)["_torch_leaf_params"] is state
        assert vars(owner)["_torch_retained"] is state.retained
        assert vars(owner)["_active_optimizers"] is state.active_optimizers
    transaction.rollback()
    assert "_torch_compat_owner" not in vars(native)
    assert "_torch_compat_owner" not in vars(target)
    assert "_torch_tensor_state" not in vars(target)
    assert get_tensor_state(native) is state


def test_tensor_state_binding_rejects_conflicting_owners_and_states():
    native = ModuleType("native_tensor_conflict")
    target = TorchNamespace(native)
    native_state = get_tensor_state(native)
    vars(target)["_torch_tensor_state"] = TorchTensorState()
    transaction = InstallTransaction("tensor-conflict")
    with pytest.raises(RuntimeError, match="conflicting tensor states"):
        bind_tensor_state(native, target, transaction)
    assert "_torch_compat_owner" not in vars(native)
    del vars(target)["_torch_tensor_state"]
    bind_tensor_state(native, target, transaction)
    with pytest.raises(RuntimeError, match="different active Torch owner"):
        bind_tensor_state(native, TorchNamespace(native), transaction)
    assert get_tensor_state(native) is native_state


def test_bound_tensor_owner_cannot_recreate_missing_local_state():
    native = ModuleType("native_missing_tensor_state")
    target = TorchNamespace(native)
    transaction = InstallTransaction("tensor-missing")
    bind_tensor_state(native, target, transaction)
    del vars(target)["_torch_tensor_state"]
    for owner in (native, target):
        with pytest.raises(RuntimeError, match="no local tensor state"):
            get_tensor_state(owner)
    assert "_torch_tensor_state" not in vars(target)


def test_tensor_state_undo_restores_entries_without_replacing_containers():
    state = TorchTensorState()
    leaf, retained, first, second, added = (object() for _ in range(5))
    state["leaf"] = leaf
    state.retained["retained"] = retained
    state.active_optimizers[:] = [first, second]
    before = snapshot_tensor_state(state)
    del state["leaf"]
    state["added"] = added
    state.retained["retained"] = added
    state.active_optimizers[:] = [second, added, first]
    transaction = InstallTransaction("tensor-entries")
    record_tensor_state_changes(transaction, state, before)
    transaction.rollback()
    assert state == {"leaf": leaf}
    assert state.retained is before["retained"]
    assert state.retained == {"retained": retained}
    assert state.active_optimizers is before["optimizers"]
    assert state.active_optimizers == [first, second]


def test_tensor_state_undo_reports_foreign_write_and_restores_other_entries():
    state = TorchTensorState()
    before = snapshot_tensor_state(state)
    state["owned"] = object()
    state.retained["owned"] = object()
    state.active_optimizers.append(object())
    transaction = InstallTransaction("tensor-foreign")
    record_tensor_state_changes(transaction, state, before)
    foreign = object()
    state["owned"] = foreign
    state.retained["foreign"] = foreign
    state.active_optimizers.append(foreign)
    with pytest.raises(TransactionConflict):
        transaction.rollback()
    assert state["owned"] is foreign
    assert state.retained == {"foreign": foreign}
    assert state.active_optimizers == [foreign]


def test_tensor_state_retry_reuses_the_staged_state_after_rollback():
    native = ModuleType("native_tensor_retry")
    target = TorchNamespace(native)
    transaction = InstallTransaction("tensor-retry")
    state = bind_tensor_state(native, target, transaction)
    before = snapshot_tensor_state(state)
    state["temporary"] = object()
    record_tensor_state_changes(transaction, state, before)
    transaction.rollback()
    assert not state
    assert "_torch_tensor_state" not in vars(native)
    retry = transaction.retry()
    assert bind_tensor_state(native, target, retry, state=state) is state
    assert get_tensor_state(native) is get_tensor_state(target) is state
