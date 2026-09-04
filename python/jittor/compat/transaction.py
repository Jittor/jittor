"""Reversible mutation ledger for compatibility installation (7.05 precursor)."""
from __future__ import absolute_import

import threading


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
        self._entries.append((target, name, old, new, undo))

    def rollback(self):
        if self.state == "committed":
            raise RuntimeError("committed transaction cannot rollback")
        with self._lock:
            for target, name, old, _new, undo in reversed(self._entries):
                if undo is not None:
                    undo()
                elif isinstance(target, dict):
                    if old is _MISSING:
                        target.pop(name, None)
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


__all__ = ["InstallTransaction"]
