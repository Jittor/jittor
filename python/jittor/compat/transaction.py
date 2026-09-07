"""Reversible mutation ledger for compatibility installation (7.05 precursor)."""
from __future__ import absolute_import

import threading

from .diagnostics import EXPECTED, swallowed


class InstallTransaction:
    """Record reversible mutations and publish them atomically under a process lock."""
    _lock = threading.RLock()

    def __init__(self, owner):
        self.owner = owner
        self.token = object()
        self._entries = []
        self.state = "open"

    def record(self, target, name, old, new, undo=None):
        if self.state != "open":
            raise RuntimeError("transaction is %s" % self.state)
        self._entries.append((target, name, old, new, undo, self.token))

    def record_undo(self, undo):
        """Register a whole-snapshot undo callback."""
        self.record({}, "__snapshot__", _MISSING, _MISSING, undo=undo)

    def record_object_diffs(self, target, before):
        """Record shallow attribute changes on an object for failure rollback."""
        after = vars(target)
        capture = getattr(type(target), "_binding_state", None)
        restore = getattr(type(target), "_restore_binding", None)
        if callable(capture) and callable(restore):
            # Namespace masks are copy-on-write, so the shallow before snapshot
            # retains the old visibility as well as the old local attributes.
            hidden_name = "_hidden_owner_names"
            old_hidden = before.get(hidden_name, ())
            new_hidden = after.get(hidden_name, ())
            names = set(before) | set(after) | set(old_hidden) | set(new_hidden)
            for name in names - {hidden_name}:
                old = (name in before, before.get(name), name in old_hidden)
                new = capture(target, name)
                if old[0] != new[0] or old[2] != new[2] or old[1] is not new[1]:
                    self._record_binding(target, name, old, [new], capture, restore)
            return
        for name in set(before) | set(after):
            old = before.get(name, _MISSING)
            new = after.get(name, _MISSING)
            if old is not new:
                self.record(target, name, old, new)

    def record_mapping_diffs(self, target, before):
        for name in before.keys() | target.keys():
            old, new = before.get(name, _MISSING), target.get(name, _MISSING)
            if old is not new:
                self.record(target, name, old, new)

    def _record_binding(self, target, name, old, expected, capture, restore):
        """Undo one local namespace slot without manufacturing a delete mask."""
        def undo():
            current = capture(target, name)
            new = expected[0]
            if (current[0] != new[0] or current[2] != new[2]
                    or (current[0] and not _matches(current[1], new[1]))):
                raise TransactionConflict(
                    "transaction owner lost %r during rollback" % name)
            restore(target, name, old)
        self.record(target, name, old, expected, undo=undo)

    def mutate_env(self, key, value, environ=None):
        import os
        env = os.environ if environ is None else environ
        old = env.get(key, _MISSING)
        normalized = str(value)
        self.record(env, key, old, normalized)
        env[key] = normalized

    def mutate_flag(self, flags, name, value):
        old = getattr(flags, name, _MISSING)
        self.record(flags, name, old, value)
        setattr(flags, name, value)

    def mutate_attr(self, target, name, value):
        capture = getattr(type(target), "_binding_state", None)
        restore = getattr(type(target), "_restore_binding", None)
        if callable(capture) and callable(restore):
            old = capture(target, name)
            expected = [old]
            self._record_binding(target, name, old, expected, capture, restore)
            try:
                setattr(target, name, value)
            finally:
                expected[0] = capture(target, name)
            return
        old = getattr(target, name, _MISSING)
        self.record(target, name, old, value)
        setattr(target, name, value)

    def mutate_path(self, paths, value, prepend=False):
        """Insert an owned path once and remove only that insertion on rollback."""
        if value in paths:
            return False
        index = 0 if prepend else len(paths)
        paths.insert(index, value)
        self.record_undo(lambda: paths.pop(index) if len(paths) > index and paths[index] == value
                         else (_raise_conflict("path entry changed externally")))
        return True

    def publish_module(self, modules, name, module):
        old = modules.get(name, _MISSING)
        if old is not _MISSING and old is not module:
            raise TransactionConflict("module %r already owned externally" % name)
        modules[name] = module
        self.record(modules, name, old, module)

    def acquire(self):
        self._lock.acquire()

    def release(self):
        self._lock.release()

    def rollback(self):
        if self.state == "committed":
            raise RuntimeError("committed transaction cannot rollback")
        with self._lock:
            conflicts = []
            for entry in reversed(self._entries):
                try:
                    self._undo(entry)
                except TransactionConflict as conflict:
                    # Keep undoing the remaining entries. Returning at the first
                    # foreign write left every *earlier* mutation of this
                    # transaction applied -- exactly the unknown mid-install
                    # state the ledger exists to prevent. The entries another
                    # actor took over are named together below instead, so the
                    # hard failure still says what was not restored.
                    conflicts.append(str(conflict))
            if conflicts:
                # "failed" rather than "open": a half-reverted ledger is still a
                # *known* state, and saying so is what lets retry() build a
                # fresh transaction and commit() refuse this one.
                self.state = "failed"
                raise TransactionConflict("; ".join(conflicts))
            self.state = "rolled_back"

    def _undo(self, entry):
        target, name, old, new, undo, owner = entry
        if undo is not None:
            undo()
            return
        current = _read(target, name)
        if not _matches(current, new):
            raise TransactionConflict(
                "transaction owner lost %r during rollback" % name
            )
        if isinstance(target, (dict, list)):
            if old is _MISSING:
                if isinstance(target, dict):
                    target.pop(name, None)
                else:
                    raise TransactionConflict(
                        "list entry %r cannot be removed safely" % name
                    )
            else:
                target[name] = old
        elif old is _MISSING:
            delattr(target, name)
        else:
            setattr(target, name, old)

    def commit(self):
        if self.state != "open":
            raise RuntimeError("transaction is %s" % self.state)
        self.state = "committed"

    def retry(self):
        if self.state not in ("rolled_back", "failed"):
            raise RuntimeError("retry requires a rolled-back transaction")
        return InstallTransaction(self.owner)


ACTIVE_TRANSACTION_KEY = "_install_transaction"


def active_transaction(context=None):
    """The transaction currently recording for ``context``, or None.

    Six installers used to inline their own copy of this lookup and five of them
    omitted the state check, so a transaction left in ``context.state`` after it
    had been committed or rolled back turned the next write into
    ``RuntimeError: transaction is <state>`` rather than a write. That matters
    beyond install: the factory and tensor owners call the same helpers from
    ``torch.zeros(device="cuda")`` at runtime, long after any ledger is closed.
    """
    if context is None:
        import jittor
        from .torch.tensor_state import compatibility_owner
        context = vars(compatibility_owner(jittor)).get("_torch_compat_install_context")
    state = getattr(context, "state", None)
    if not isinstance(state, dict):
        return None
    transaction = state.get(ACTIVE_TRANSACTION_KEY)
    if getattr(transaction, "state", None) != "open":
        return None
    return transaction


def set_flag(flags, name, value, context=None):
    """Write a Jittor flag, reversibly while an install is recording."""
    transaction = active_transaction(context)
    if transaction is None:
        setattr(flags, name, value)
    else:
        transaction.mutate_flag(flags, name, value)


def set_env(key, value, context=None, environ=None):
    """Write an environment variable, reversibly while an install is recording."""
    transaction = active_transaction(context)
    if transaction is not None:
        transaction.mutate_env(key, value, environ=environ)
        return
    import os
    env = os.environ if environ is None else environ
    # str() unconditionally, matching mutate_env: the recorded owner value and
    # the direct write have to be the same text or rollback reports a conflict
    # against a value it wrote itself (the integer-valued rank variables).
    env[key] = str(value)


def set_attr(target, name, value, context=None):
    """Write an object attribute, reversibly while an install is recording."""
    transaction = active_transaction(context)
    if transaction is None:
        setattr(target, name, value)
    else:
        transaction.mutate_attr(target, name, value)


class _Missing:
    pass


_MISSING = _Missing()


def _raise_conflict(message):
    raise TransactionConflict(message)


class TransactionConflict(RuntimeError):
    """Raised when rollback would overwrite a value owned by another actor."""


class ActivationTransaction(InstallTransaction):
    """Transaction protocol for shim activation path/module/flag mutations."""

    pass


def _read(target, name):
    if isinstance(target, (dict, list)):
        try:
            return target[name]
        except (KeyError, IndexError):
            return _MISSING
    return getattr(target, name, _MISSING)


def _matches(current, expected):
    if current is expected:
        return True
    if current is _MISSING or expected is _MISSING:
        return False
    try:
        result = current == expected
        return bool(result) if isinstance(result, bool) else False
    except EXPECTED as exc:
        swallowed("transaction.py _matches: result = current == expected", exc,
                  "the recorded value is treated as changed by someone else, so "
                  "rollback raises TransactionConflict instead of restoring it")
        return False


__all__ = [
    "ACTIVE_TRANSACTION_KEY",
    "ActivationTransaction",
    "InstallTransaction",
    "TransactionConflict",
    "active_transaction",
    "set_attr",
    "set_env",
    "set_flag",
]
