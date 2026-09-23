import sys
import types
import pytest
from jittor.compat.transaction import InstallTransaction, TransactionConflict


def test_rollback_preserves_conflict_without_importing_tensor_cache(monkeypatch):
    """Cleanup must not invoke a foreign finder to load unused tensor caches."""
    monkeypatch.delitem(sys.modules, "jittor.compat.torch.tensor_state", raising=False)

    class RejectTensorImport:
        def find_spec(self, fullname, path=None, target=None):
            if fullname.startswith("jittor.compat.torch"):
                raise AssertionError("rollback attempted a new tensor-state import")
            return None

    target = types.SimpleNamespace(value="original")
    tx = InstallTransaction("vllm.rollback-import")
    tx.record(target, "value", "original", "owned")
    target.value = "external"
    original_meta_path = list(sys.meta_path)
    try:
        sys.meta_path.insert(0, RejectTensorImport())
        with pytest.raises(TransactionConflict, match="lost.*value"):
            tx.rollback()
        assert target.value == "external"
        assert tx.state == "failed"
    finally:
        sys.meta_path[:] = original_meta_path


def test_rollback_clears_already_loaded_tensor_cache(monkeypatch):
    module = types.ModuleType("jittor.compat.torch.tensor_state")
    cached_bindings = {"owner": object()}
    module._clear_resolution_caches = cached_bindings.clear
    monkeypatch.setitem(sys.modules, module.__name__, module)
    tx = InstallTransaction("vllm.rollback-cache")
    tx.rollback()
    assert cached_bindings == {}
    assert tx.state == "rolled_back"


def test_vllm_arming_finder_rollback_rejects_an_external_replacement():
    """The one failure the ledger exists to surface must not be the one it hides.

    This undo used to be ``remove(f) if f in sys.meta_path else None``. When
    another actor had already dropped or replaced the entry, rollback reported
    success and left their finder installed -- silently, which is the shape of
    every defect in this layer that took a week to find.
    """
    from jittor_adapters.vllm import bootstrap as vllm

    original_meta_path = list(sys.meta_path)
    original_installed = vllm._installed
    try:
        vllm._installed = False
        sys.meta_path[:] = [
            finder for finder in sys.meta_path
            if not isinstance(finder, vllm._ArmOnFirstImport)
        ]
        tx = InstallTransaction("vllm.register")
        vllm.arm(transaction=tx)
        finder = sys.meta_path[0]
        assert isinstance(finder, vllm._ArmOnFirstImport)

        replacement = object()
        sys.meta_path[0] = replacement
        with pytest.raises(TransactionConflict, match="moved or replaced"):
            tx.rollback()
        assert sys.meta_path[0] is replacement
    finally:
        sys.meta_path[:] = original_meta_path
        vllm._installed = original_installed
def test_vllm_arming_finder_rollback_removes_the_entry_it_owns():
    from jittor_adapters.vllm import bootstrap as vllm

    original_meta_path = list(sys.meta_path)
    original_installed = vllm._installed
    try:
        vllm._installed = False
        sys.meta_path[:] = [
            finder for finder in sys.meta_path
            if not isinstance(finder, vllm._ArmOnFirstImport)
        ]
        tx = InstallTransaction("vllm.register")
        vllm.arm(transaction=tx)
        finder = sys.meta_path[0]
        tx.rollback()
        assert finder not in sys.meta_path
    finally:
        sys.meta_path[:] = original_meta_path
        vllm._installed = original_installed
