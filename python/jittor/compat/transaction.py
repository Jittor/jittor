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
        for name in set(before) | set(after):
            old = before.get(name, _MISSING)
            new = after.get(name, _MISSING)
            if old is not new:
                self.record(target, name, old, new)

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
            for target, name, old, new, undo, owner in reversed(self._entries):
                if undo is not None:
                    undo()
                else:
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
                    else:
                        if old is _MISSING:
                            delattr(target, name)
                        else:
                            setattr(target, name, old)
            self.state = "rolled_back"

    def commit(self):
        if self.state != "open":
            raise RuntimeError("transaction is %s" % self.state)
        self.state = "committed"

    def retry(self):
        if self.state not in ("rolled_back", "failed"):
            raise RuntimeError("retry requires a rolled-back transaction")
        return InstallTransaction(self.owner)


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


__all__ = ["InstallTransaction", "ActivationTransaction", "TransactionConflict"]
