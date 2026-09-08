from types import SimpleNamespace
from types import ModuleType
import gc
import weakref
import pytest

from jittor.compat.torch.tensor_state import (
    TorchTensorState, get_tensor_state, compatibility_owner, bind_tensor_state,
    snapshot_tensor_state, record_tensor_state_changes,
    latest_optimizer,
)
from jittor.compat.torch.namespace import TorchNamespace
from jittor.compat.transaction import InstallTransaction, TransactionConflict
from jittor._runtime.state import RuntimeState, RuntimeContext


def _module(name="tensor_state_test", **fields):
    module = ModuleType(name)
    module.runtime = RuntimeState(RuntimeContext(SimpleNamespace()))
    vars(module).update(fields)
    return module


def _assert_no_state_aliases(module):
    assert not ({"_torch_tensor_state", "_torch_leaf_params", "_torch_retained",
                 "_active_optimizers", "_current_optimizer"} & vars(module).keys())


def test_tensor_state_adopts_optimizer_registry_without_module_aliases():
    legacy = []
    module = _module(_torch_leaf_params={}, _active_optimizers=legacy)

    state = get_tensor_state(module)

    assert isinstance(state, TorchTensorState)
    assert state.active_optimizers is legacy
    state.active_optimizers.append("optimizer")
    assert legacy == ["optimizer"]
    _assert_no_state_aliases(module)
    assert module.runtime.service_state("jittor.torch.tensor_states")[module] is state


def test_tensor_state_adopts_legacy_leaves_without_publishing_aliases():
    module = _module(_torch_leaf_params={"leaf": object()})

    state = get_tensor_state(module)

    assert state.leaf_params is state
    assert "leaf" in state
    _assert_no_state_aliases(module)


def test_tensor_state_migrates_existing_retained_entries():
    retained = {"retained": object()}
    module = _module(_torch_leaf_params={}, _torch_retained=retained)

    state = get_tensor_state(module)

    assert state.retained == retained
    assert state.retained is not retained
    _assert_no_state_aliases(module)


def test_tensor_state_reuses_explicit_owner_over_legacy_alias():
    state = TorchTensorState()
    legacy = {"stale": object()}
    module = _module(
        _torch_tensor_state=state,
        _torch_leaf_params=legacy,
    )

    assert get_tensor_state(module) is state
    _assert_no_state_aliases(module)
    assert "stale" not in state


def test_tensor_state_does_not_duplicate_native_requires_grad_state():
    module = _module()
    state = get_tensor_state(module)
    assert not hasattr(state, "requires_grad")
    tensor = object()
    state.leaf_params[id(tensor)] = tensor
    state.retained[id(tensor)] = tensor
    assert get_tensor_state(module) is state
    assert state.leaf_params[id(tensor)] is tensor
    assert state.retained[id(tensor)] is tensor
    _assert_no_state_aliases(module)


def test_tensor_state_releases_values_removed_from_backward_registries():
    module = _module()
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
    native = _module("native_tensor_owner")
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
        _assert_no_state_aliases(owner)
    transaction.rollback()
    assert "_torch_compat_owner" not in vars(native)
    assert "_torch_compat_owner" not in vars(target)
    assert "_torch_tensor_state" not in vars(target)
    assert get_tensor_state(native) is state


def test_independent_binding_does_not_publish_state_on_native_module():
    native = _module("untouched_native_owner")
    before = vars(native).copy()
    target = TorchNamespace(native)
    transaction = InstallTransaction("local-owner-only")
    state = bind_tensor_state(native, target, transaction)
    assert get_tensor_state(native) is get_tensor_state(target) is state
    assert vars(native) == before
    assert "_torch_compat_owner" not in vars(target)
    transaction.rollback()
    assert compatibility_owner(native) is native
    assert vars(native) == before


def test_legacy_binding_keeps_runtime_state_off_the_root():
    native = _module("legacy_native_owner")
    before = vars(native).copy()
    transaction = InstallTransaction("legacy-owner")
    state = bind_tensor_state(native, native, transaction)
    assert get_tensor_state(native) is state
    assert vars(native) == before
    transaction.rollback()
    assert not native.runtime.service_state("jittor.torch.tensor_states")


def test_concurrent_first_lookup_publishes_one_state(monkeypatch):
    import threading
    from jittor.compat.torch import tensor_state as owner
    native = _module("concurrent_first_owner")
    entered, release = threading.Event(), threading.Event()
    original = owner._adopt_legacy_state
    created, results = [], []
    def construct(module):
        value = original(module)
        created.append(value)
        entered.set()
        assert release.wait(3)
        return value
    monkeypatch.setattr(owner, "_adopt_legacy_state", construct)
    first = threading.Thread(target=lambda: results.append(get_tensor_state(native)))
    second = threading.Thread(target=lambda: results.append(get_tensor_state(native)))
    first.start()
    assert entered.wait(3)
    second.start()
    release.set()
    first.join(3)
    second.join(3)
    assert not first.is_alive() and not second.is_alive()
    assert len(created) == 1 and len(results) == 2
    assert results[0] is results[1] is get_tensor_state(native)


def test_state_rollback_rejects_an_equal_but_foreign_state():
    native = _module("foreign_tensor_state")
    transaction = InstallTransaction("state-owner-conflict")
    state = bind_tensor_state(native, native, transaction)
    table = native.runtime.service_state("jittor.torch.tensor_states")
    foreign = TorchTensorState()
    assert foreign == state and foreign is not state
    table[native] = foreign
    with pytest.raises(TransactionConflict, match="tensor-state owner changed"):
        transaction.rollback()
    assert table[native] is foreign


def test_owner_lookup_does_not_keep_an_unloaded_frontend_alive():
    native = _module("unloadable_native_owner")
    target = TorchNamespace(native)
    transaction = InstallTransaction("unloadable-owner")
    bind_tensor_state(native, target, transaction)
    transaction.commit()
    reference = weakref.ref(target)
    del target, transaction
    gc.collect()
    assert reference() is None
    assert compatibility_owner(native) is native


def test_latest_optimizer_lookup_does_not_prevent_collection():
    module = _module("optimizer_owner")
    state = get_tensor_state(module)
    class Optimizer:
        pass
    first, last = Optimizer(), Optimizer()
    state.active_optimizers[:] = [weakref.ref(first), weakref.ref(last)]
    assert latest_optimizer(module) is last
    reference = weakref.ref(last)
    del last
    gc.collect()
    assert reference() is None
    assert latest_optimizer(module) is first


def test_tensor_state_binding_rejects_conflicting_owners_and_states():
    native = _module("native_tensor_conflict")
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
    native = _module("native_missing_tensor_state")
    target = TorchNamespace(native)
    transaction = InstallTransaction("tensor-missing")
    bind_tensor_state(native, target, transaction)
    del target.runtime.service_state("jittor.torch.tensor_states")[target]
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
    native = _module("native_tensor_retry")
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
