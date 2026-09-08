import sys
import pytest
from jittor.compat.transaction import InstallTransaction, TransactionConflict

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

