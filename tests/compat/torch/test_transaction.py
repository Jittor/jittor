import threading
import types

import pytest

from jittor.compat.transaction import InstallTransaction, TransactionConflict
from jittor.compat.torch.installers.core import _set_install_flag
from jittor.compat.torch.installers.utilities import _mutate_import


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


def test_core_install_flag_mutation_rolls_back_on_failure():
    flags = types.SimpleNamespace(use_cuda=0)
    tx = InstallTransaction("core.install")
    ctx = types.SimpleNamespace(state={"_install_transaction": tx})
    original = __import__("jittor").flags
    saved = original.use_cuda
    try:
        original.use_cuda = flags.use_cuda
        _set_install_flag(ctx, "use_cuda", 1)
        assert original.use_cuda == 1
        tx.rollback()
        assert original.use_cuda == 0
    finally:
        original.use_cuda = saved


def test_utilities_import_hook_rolls_back_and_detects_external_replacement():
    import builtins
    import jittor

    tx = InstallTransaction("utilities.install")
    context = types.SimpleNamespace(state={"_install_transaction": tx})
    previous_context = getattr(jittor, "_torch_compat_install_context", None)
    original_import = builtins.__import__

    def replacement(*args, **kwargs):
        return original_import(*args, **kwargs)

    try:
        jittor._torch_compat_install_context = context
        _mutate_import(replacement, builtins)
        assert builtins.__import__ is replacement
        tx.rollback()
        assert builtins.__import__ is original_import

        tx = InstallTransaction("utilities.install.conflict")
        context.state["_install_transaction"] = tx
        _mutate_import(replacement, builtins)
        builtins.__import__ = original_import
        with pytest.raises(TransactionConflict, match="owner lost"):
            tx.rollback()
        assert builtins.__import__ is original_import
    finally:
        builtins.__import__ = original_import
        if previous_context is None:
            delattr(jittor, "_torch_compat_install_context")
        else:
            jittor._torch_compat_install_context = previous_context


def test_environment_mutation_records_the_normalized_string_value():
    env = {}
    tx = InstallTransaction("env-owner")
    tx.mutate_env("RANK", 1, environ=env)
    assert env["RANK"] == "1"
    tx.rollback()
    assert "RANK" not in env
