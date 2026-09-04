import threading
import types

import pytest

from jittor.compat.transaction import InstallTransaction, TransactionConflict


def test_transaction_rolls_back_module_env_flags_and_meta_path_in_reverse_order():
    module = types.SimpleNamespace(value="old")
    env = {"MODE": "old"}
    flags = {"amp_reg": 0}
    meta_path = ["finder"]
    tx = InstallTransaction("test")
    tx.record(module, "value", module.value, "new")
    tx.record(env, "MODE", env["MODE"], "new")
    tx.record(flags, "amp_reg", flags["amp_reg"], 3)
    tx.record(meta_path, 0, meta_path[0], "replacement", undo=lambda: meta_path.__setitem__(0, "finder"))
    module.value, env["MODE"], flags["amp_reg"], meta_path[0] = "new", "new", 3, "replacement"
    tx.rollback()
    assert (module.value, env["MODE"], flags["amp_reg"], meta_path) == ("old", "old", 0, ["finder"])
    assert tx.retry().owner == "test"


def test_transaction_commit_and_process_lock_are_serializable():
    tx = InstallTransaction("owner")
    tx.commit()
    assert tx.state == "committed"
    assert isinstance(InstallTransaction._lock, type(threading.RLock()))


def test_transaction_refuses_to_overwrite_an_external_attribute_change():
    module = types.SimpleNamespace(value="old")
    tx = InstallTransaction("owner")
    tx.record(module, "value", "old", "transaction")
    module.value = "external"
    with pytest.raises(TransactionConflict, match="owner lost"):
        tx.rollback()
    assert module.value == "external"
