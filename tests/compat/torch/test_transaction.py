import sys
import os
import threading
import types

import pytest

from jittor.compat.transaction import (
    ActivationTransaction,
    InstallTransaction,
    TransactionConflict,
    set_attr,
    set_env,
    set_flag,
)
from jittor.compat.torch.installers.core import _set_install_flag
from jittor.compat.torch.installers.utilities import _mutate_import

from _helpers.install_lock import install_lock_is_free


def test_real_process_environment_rolls_back_added_and_existing_keys(monkeypatch):
    key = "JITTOR_TEST_TRANSACTION_MODE"
    monkeypatch.delenv(key, raising=False)
    transaction = InstallTransaction("process-environment")
    transaction.mutate_env(key, "independent")
    transaction.rollback()
    assert key not in os.environ
    monkeypatch.setenv(key, "legacy")
    transaction = InstallTransaction("existing-process-environment")
    transaction.mutate_env(key, "independent")
    transaction.rollback()
    assert os.environ[key] == "legacy"


def test_failed_target_install_replays_rolled_back_steps_with_same_state(monkeypatch):
    from jittor.compat import torch as installer
    from jittor.compat.torch.namespace import TorchNamespace
    from jittor.compat.torch.tensor_state import get_tensor_state

    backend = types.ModuleType("test-native-backend")
    target = TorchNamespace(backend)
    marker = object()
    calls = []
    fail = [True]

    def first(ctx):
        calls.append("first")
        ctx.target_namespace.local_api = marker
        get_tensor_state(ctx.native_backend).leaf_params[17] = marker

    def second(ctx):
        if fail[0]:
            raise ValueError("injected target install failure")

    monkeypatch.setattr(installer, "_REQUIRED_STEPS", (("first", first), ("second", second)))
    monkeypatch.setattr(installer, "_OPTIONAL_STEPS", ())
    with pytest.raises(installer.InstallStepError, match="injected target install failure"):
        installer.install(target)
    ctx = vars(target)["_torch_compat_install_context"]
    state = ctx.state["_tensor_state"]
    assert not state
    assert "local_api" not in vars(target)
    assert "_torch_compat_owner" not in vars(backend)
    assert ctx.markers.get("first") != "complete"

    fail[0] = False
    assert installer.install(target) is target
    assert calls == ["first", "first"]
    assert get_tensor_state(backend) is state
    assert state[17] is marker
    assert target.local_api is marker


def _context(transaction=None):
    """An install context stand-in carrying only the ledger handle."""
    state = {} if transaction is None else {"_install_transaction": transaction}
    return types.SimpleNamespace(state=state)


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

    from jittor.compat.torch.namespace import TorchNamespace
    namespace = TorchNamespace(types.SimpleNamespace(value="native"))
    tx = InstallTransaction("namespace-owner")
    tx.mutate_attr(namespace, "value", "transaction")
    del namespace.value
    with pytest.raises(TransactionConflict, match="owner lost"):
        tx.rollback()
    assert not hasattr(namespace, "value")
    assert namespace.owner.value == "native"


def test_conflicting_rollback_still_reverts_everything_it_still_owns():
    """One foreign write must not strand the transaction's other mutations.

    Rollback walks newest first, so returning at the first conflict left every
    *earlier* mutation applied -- a process half-way through an install, which
    is the state the ledger exists to rule out. Only the entries another actor
    took over stay, and the hard failure names them.
    """
    module = types.SimpleNamespace(first="old-first", second="old-second")
    env = {"MODE": "old"}
    tx = InstallTransaction("owner")
    tx.mutate_attr(module, "first", "ours-first")
    tx.mutate_env("MODE", "ours", environ=env)
    tx.mutate_attr(module, "second", "ours-second")
    module.second = "someone else"

    with pytest.raises(TransactionConflict, match="owner lost 'second'"):
        tx.rollback()

    assert module.first == "old-first"
    assert env["MODE"] == "old"
    assert module.second == "someone else"
    # A half-reverted ledger is a known state, not an open one: commit() has to
    # refuse it and retry() has to accept it.
    assert tx.state == "failed"
    with pytest.raises(RuntimeError, match="transaction is failed"):
        tx.commit()
    assert tx.retry().state == "open"


def test_conflicting_rollback_names_every_entry_it_could_not_restore():
    module = types.SimpleNamespace(first="old-first", second="old-second")
    tx = InstallTransaction("owner")
    tx.mutate_attr(module, "first", "ours-first")
    tx.mutate_attr(module, "second", "ours-second")
    module.first = "stolen-first"
    module.second = "stolen-second"

    with pytest.raises(TransactionConflict) as raised:
        tx.rollback()
    assert "'first'" in str(raised.value) and "'second'" in str(raised.value)


def test_core_install_flag_mutation_rolls_back_on_failure():
    flags = types.SimpleNamespace(use_cuda=0)
    tx = InstallTransaction("core.install")
    ctx = types.SimpleNamespace(
        state={"_install_transaction": tx},
        native_backend=types.SimpleNamespace(flags=flags),
    )
    _set_install_flag(ctx, "use_cuda", 1)
    assert flags.use_cuda == 1
    tx.rollback()
    assert flags.use_cuda == 0


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


def test_shared_write_helpers_record_flag_env_and_attribute_mutations():
    """One owner for every install-time write, so one rollback path covers them."""
    flags = types.SimpleNamespace(use_cuda=0)
    env = {}
    target = types.SimpleNamespace()
    tx = InstallTransaction("owner")
    context = _context(tx)

    set_flag(flags, "use_cuda", 1, context=context)
    set_env("JT_NCCL_RANK", 3, context=context, environ=env)
    set_attr(target, "__import__", "ours", context=context)
    assert (flags.use_cuda, env["JT_NCCL_RANK"], target.__import__) == (1, "3", "ours")

    tx.rollback()
    assert flags.use_cuda == 0
    assert "JT_NCCL_RANK" not in env
    assert not hasattr(target, "__import__")

    from jittor.compat.torch.namespace import TorchNamespace
    owner = types.SimpleNamespace(value="native", hidden="native-hidden")
    namespace = TorchNamespace(owner)
    del namespace.hidden
    tx = InstallTransaction("namespace-owner")
    context = _context(tx)
    set_attr(namespace, "value", "local", context=context)
    set_attr(namespace, "hidden", "visible", context=context)
    set_attr(namespace, "added", "temporary", context=context)
    tx.rollback()
    assert "value" not in vars(namespace)
    assert namespace.value == "native"
    owner.value = "updated-native"
    assert namespace.value == "updated-native"
    assert not hasattr(namespace, "hidden")
    assert "added" not in vars(namespace)
    owner.added = "new-native"
    assert namespace.added == "new-native"


@pytest.mark.parametrize("closed", ("committed", "rolled_back"))
def test_installer_writes_ignore_a_ledger_that_has_already_closed(closed):
    """A closed ledger must not turn a later write into a RuntimeError.

    These helpers do not run only at install time: ``_set_use_cuda`` is reached
    from ``torch.zeros(device="cuda")`` and ``_mutate_import`` from the optional
    integration steps. Five of the six inlined lookups they replaced took
    whatever transaction sat in ``context.state`` without checking its state, and
    ``record()`` refuses one that is no longer open -- so a ledger left behind by
    a failed install turned every later such write into
    ``RuntimeError: transaction is rolled_back`` instead of a write.
    """
    import builtins
    import jittor

    tx = InstallTransaction("closed")
    if closed == "committed":
        tx.commit()
    else:
        tx.rollback()

    previous = getattr(jittor, "_torch_compat_install_context", None)
    original_import = builtins.__import__

    def replacement(*args, **kwargs):
        return original_import(*args, **kwargs)

    try:
        jittor._torch_compat_install_context = _context(tx)
        _mutate_import(replacement, builtins)
        assert builtins.__import__ is replacement
    finally:
        builtins.__import__ = original_import
        if previous is None:
            delattr(jittor, "_torch_compat_install_context")
        else:
            jittor._torch_compat_install_context = previous


def test_shared_write_helpers_ignore_a_closed_ledger_for_flags_too():
    flags = types.SimpleNamespace(use_cuda=0)
    tx = InstallTransaction("closed")
    tx.commit()
    set_flag(flags, "use_cuda", 1, context=_context(tx))
    assert flags.use_cuda == 1


def test_direct_environment_write_normalizes_like_the_recorded_one():
    """The two paths have to agree on the text, or rollback finds a conflict.

    ``mutate_env`` stores ``str(value)``; the direct write it falls back to used
    to store the raw object for some callers, and the integer-valued rank
    variables then failed their own owner check.
    """
    env = {}
    set_env("JT_NCCL_RANK", 3, context=_context(), environ=env)
    assert env["JT_NCCL_RANK"] == "3"


def test_concurrent_external_replacement_is_reported_not_overwritten():
    """The process lock only orders the actors that ask for it.

    A thread that writes ``builtins.__import__`` or a flag without taking the
    install lock still races an open ledger. Serialising installs cannot prevent
    that, so the owner check at rollback is the whole defence: report the foreign
    value, never restore over it.
    """
    module = types.SimpleNamespace(hook="original")
    tx = InstallTransaction("install")
    tx.acquire()
    try:
        set_attr(module, "hook", "ours", context=_context(tx))
        assert not install_lock_is_free(timeout=0.5), (
            "an open install transaction has to exclude other threads"
        )

        replaced = threading.Event()

        def foreign_writer():
            module.hook = "another library"
            replaced.set()

        thread = threading.Thread(target=foreign_writer)
        thread.start()
        thread.join(5.0)
        assert replaced.is_set()

        with pytest.raises(TransactionConflict, match="owner lost 'hook'"):
            tx.rollback()
        assert module.hook == "another library"
        assert tx.state == "failed"
    finally:
        tx.release()
    assert install_lock_is_free()


def test_environment_mutation_records_the_normalized_string_value():
    env = {}
    tx = InstallTransaction("env-owner")
    tx.mutate_env("RANK", 1, environ=env)
    assert env["RANK"] == "1"
    tx.rollback()
    assert "RANK" not in env


def test_activation_transaction_path_and_module_owner_conflicts():
    paths = ["stdlib"]
    modules = {}
    tx = ActivationTransaction("shim.activate")
    tx.mutate_path(paths, "shim", prepend=True)
    tx.publish_module(modules, "torch", object())
    tx.rollback()
    assert paths == ["stdlib"]
    assert modules == {}


def test_module_patch_finder_rollback_is_owner_aware():
    from jittor.compat import module_patcher

    original_path = list(module_patcher.sys.meta_path)
    original_finder = module_patcher._FINDER
    original_registry = {
        path: list(callbacks)
        for path, callbacks in module_patcher._REGISTRY.items()
    }
    original_loaded = set(module_patcher._ENTRY_POINTS_LOADED)
    try:
        module_patcher._FINDER = None
        module_patcher._REGISTRY.clear()
        module_patcher._ENTRY_POINTS_LOADED.clear()
        tx = InstallTransaction("module-patcher")
        module_patcher.install_module_patches(
            load_entry_points=False, transaction=tx
        )
        created = module_patcher._FINDER
        assert created in module_patcher.sys.meta_path
        tx.rollback()
        assert created not in module_patcher.sys.meta_path

        tx = InstallTransaction("module-patcher-conflict")
        module_patcher.install_module_patches(
            load_entry_points=False, transaction=tx
        )
        created = module_patcher._FINDER
        external = object()
        index = module_patcher.sys.meta_path.index(created)
        module_patcher.sys.meta_path[index] = external
        with pytest.raises(TransactionConflict, match="finder replaced"):
            tx.rollback()
        assert module_patcher.sys.meta_path[index] is external
    finally:
        module_patcher.sys.meta_path[:] = original_path
        module_patcher._FINDER = original_finder
        module_patcher._REGISTRY.clear()
        module_patcher._REGISTRY.update(
            {path: list(callbacks) for path, callbacks in original_registry.items()}
        )
        module_patcher._ENTRY_POINTS_LOADED.clear()
        module_patcher._ENTRY_POINTS_LOADED.update(original_loaded)


def test_permissive_finder_rollback_preserves_external_allowlist_changes():
    from jittor.compat import permissive

    finder = permissive._PermissiveFinder("torch._test", ("torch._test.keep",))
    meta_path = [finder]
    tx = InstallTransaction("permissive")
    permissive.install_permissive_package(
        "torch._test", meta_path, allow=("torch._test.added",), transaction=tx
    )
    finder.allow.add("torch._test.external")
    with pytest.raises(TransactionConflict, match="allowlist changed"):
        tx.rollback()
    assert "torch._test.external" in finder.allow


def test_permissive_finder_rollback_rejects_external_reordering():
    from jittor.compat import permissive

    meta_path = ["sentinel"]
    tx = InstallTransaction("permissive")
    permissive.install_permissive_package(
        "torch._test2", meta_path, allow=(), transaction=tx
    )
    finder = meta_path[0]
    meta_path[:] = ["sentinel", finder]
    with pytest.raises(TransactionConflict, match="moved or replaced"):
        tx.rollback()


def test_vllm_arming_finder_rollback_rejects_an_external_replacement():
    """The one failure the ledger exists to surface must not be the one it hides.

    This undo used to be ``remove(f) if f in sys.meta_path else None``. When
    another actor had already dropped or replaced the entry, rollback reported
    success and left their finder installed -- silently, which is the shape of
    every defect in this layer that took a week to find.
    """
    from jittor.compat import vllm

    original_meta_path = list(sys.meta_path)
    original_installed = vllm._installed
    try:
        vllm._installed = False
        sys.meta_path[:] = [
            finder for finder in sys.meta_path
            if not isinstance(finder, vllm._ArmOnFirstImport)
        ]
        tx = InstallTransaction("vllm.register")
        vllm.register(transaction=tx)
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
    from jittor.compat import vllm

    original_meta_path = list(sys.meta_path)
    original_installed = vllm._installed
    try:
        vllm._installed = False
        sys.meta_path[:] = [
            finder for finder in sys.meta_path
            if not isinstance(finder, vllm._ArmOnFirstImport)
        ]
        tx = InstallTransaction("vllm.register")
        vllm.register(transaction=tx)
        finder = sys.meta_path[0]
        tx.rollback()
        assert finder not in sys.meta_path
    finally:
        sys.meta_path[:] = original_meta_path
        vllm._installed = original_installed


def test_transaction_records_module_attribute_diffs_for_failure_rollback():
    module = types.SimpleNamespace(existing="old")
    before = dict(vars(module))
    tx = InstallTransaction("module-owner")
    module.existing = "new"
    module.added = 7
    tx.record_object_diffs(module, before)
    tx.rollback()
    assert vars(module) == {"existing": "old"}

    from jittor.compat.torch.namespace import TorchNamespace
    owner = types.SimpleNamespace(existing="native", hidden="native-hidden", deleted="native-deleted")
    namespace = TorchNamespace(owner)
    del namespace.hidden
    before = dict(vars(namespace))
    namespace.existing = "local"
    namespace.hidden = "visible"
    namespace.added = 7
    del namespace.deleted
    tx = InstallTransaction("namespace-diffs")
    tx.record_object_diffs(namespace, before)
    tx.rollback()
    assert "existing" not in vars(namespace)
    assert namespace.existing == "native"
    assert not hasattr(namespace, "hidden")
    assert namespace.deleted == "native-deleted"
    assert "added" not in vars(namespace)
    owner.added = "native-added"
    assert namespace.added == "native-added"


class _RaisesOnCompare:
    """A recorded value whose ``__eq__`` fails, as a Var's can."""

    def __init__(self, error):
        self._error = error

    def __eq__(self, other):
        raise self._error


def _rollback_over_a_failing_comparison(error):
    """Roll back a mutation whose owner check has to compare two of these."""
    module = types.SimpleNamespace(value="old")
    tx = InstallTransaction("owner")
    tx.record(module, "value", "old", _RaisesOnCompare(error))
    # A distinct instance, so the identity fast path cannot answer and the
    # owner check reaches the comparison.
    module.value = _RaisesOnCompare(error)
    tx.rollback()


def test_owner_check_treats_a_declared_comparison_failure_as_an_external_change():
    with pytest.raises(TransactionConflict, match="owner lost"):
        _rollback_over_a_failing_comparison(TypeError("no ordering"))


def test_owner_check_lets_this_layers_own_bugs_surface():
    """A NameError here is a typo in the ledger, not evidence of a foreign write.

    The handler used to be ``except Exception``, which turned any bug inside the
    comparison into "the value no longer matches" and reported it as a
    TransactionConflict -- pointing the reader at an imaginary external writer.
    """
    with pytest.raises(NameError):
        _rollback_over_a_failing_comparison(NameError("typo in the ledger"))
