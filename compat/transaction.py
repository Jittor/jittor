"""Reversible mutation ledger for compatibility installation (7.05 precursor)."""
from __future__ import absolute_import

import threading
from contextlib import contextmanager
from functools import wraps
from collections.abc import MutableMapping

from .diagnostics import EXPECTED, swallowed
_ACTIVE_TRANSACTIONS = threading.local()


class InstallTransaction:
    """Record reversible mutations and publish them atomically under a process lock."""
    _lock = threading.RLock()

    def __init__(self, owner):
        self.owner = owner
        self.token = object()
        self._entries = []
        self._covered_slots = set()
        self.state = "open"

    def record(self, target, name, old, new, undo=None):
        if self.state != "open":
            raise RuntimeError("transaction is %s" % self.state)
        self._entries.append((target, name, old, new, undo, self.token))
        self._covered_slots.add((id(target), name))

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
                if (id(target), name) in self._covered_slots:
                    continue
                old = (name in before, before.get(name), name in old_hidden)
                new = capture(target, name)
                if old[0] != new[0] or old[2] != new[2] or old[1] is not new[1]:
                    self._record_binding(target, name, old, [new], capture, restore)
            return
        for name in set(before) | set(after):
            if (id(target), name) in self._covered_slots:
                continue
            old = before.get(name, _MISSING)
            new = after.get(name, _MISSING)
            if old is not new:
                self.record(target, name, old, new)

    def record_mapping_diffs(self, target, before):
        for name in before.keys() | target.keys():
            if (id(target), name) in self._covered_slots:
                continue
            old, new = before.get(name, _MISSING), target.get(name, _MISSING)
            if old is not new:
                self.record(target, name, old, new)

    def adopt(self, child):
        """Transfer a successful child's undo ledger before either commit."""
        if self.state != "open" or child.state != "open" or child is self:
            raise RuntimeError("adoption requires distinct open transactions")
        self._entries.extend(child._entries)
        self._covered_slots.update(child._covered_slots)
        child._entries.clear()
        child._covered_slots.clear()

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
        stack = getattr(_ACTIVE_TRANSACTIONS, "stack", None)
        if stack is None:
            stack = _ACTIVE_TRANSACTIONS.stack = []
        stack.append(self)

    def release(self):
        stack = getattr(_ACTIVE_TRANSACTIONS, "stack", [])
        for index in range(len(stack)-1, -1, -1):
            if stack[index] is self:
                stack.pop(index)
                break
        self._lock.release()

    def rollback(self):
        if self.state == "committed":
            raise RuntimeError("committed transaction cannot rollback")
        with self._lock:
            conflicts = []
            errors = []
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
                except Exception as error:
                    # Finish independent cleanup, then preserve an actual
                    # implementation exception instead of calling it a foreign write.
                    errors.append(error)
            if errors:
                self.state = "failed"
                raise errors[0]
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
        if isinstance(target, (MutableMapping, list)):
            if old is _MISSING:
                if isinstance(target, MutableMapping):
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
_HOOK_LOCAL = threading.local()
_RUNTIME_HOOKS = {}
_NO_REPLACEMENT = object()


def current_runtime_hook():
    return getattr(_HOOK_LOCAL, "transaction", None)


def current_transaction():
    """Current recording scope without importing Jittor or resolving an owner."""
    for transaction in reversed(getattr(_ACTIVE_TRANSACTIONS, "stack", ())):
        if transaction.state == "open":
            return transaction
    return None


class RuntimeHook(InstallTransaction):
    """A committed hook retains its undo ledger until its owner releases it."""
    def commit(self):
        if self.state != "open":
            raise RuntimeError("runtime hook is %s" % self.state)
        self.state = "active"

    def rollback(self):
        if self.state == "rolled_back":
            return
        super().rollback()
        self._entries.clear()
        hooks = _RUNTIME_HOOKS.get(self.owner)
        if hooks is not None:
            hooks[:] = [hook for hook in hooks if hook is not self]
            if not hooks:
                _RUNTIME_HOOKS.pop(self.owner, None)

    def mutate_attr(self, target, name, value):
        if callable(getattr(type(target), "_binding_state", None)):
            return super().mutate_attr(target, name, value)
        namespace = vars(target)
        old = namespace.get(name, _MISSING)
        expected = [old]
        def undo():
            if vars(target).get(name, _MISSING) is not expected[0]:
                raise TransactionConflict("runtime hook lost attribute %r" % name)
            if expected[0] is old:
                return
            if old is _MISSING:
                delattr(target, name)
            else:
                setattr(target, name, old)
        self.record(target, name, old, value, undo=undo)
        try:
            setattr(target, name, value)
        finally:
            expected[0] = vars(target).get(name, _MISSING)

    def replace_module(self, modules, name, value, expected=_NO_REPLACEMENT):
        # expected is supplied explicitly by callers replacing an existing slot.
        old = modules.get(name, _MISSING)
        if old is not _MISSING and old is not expected and old is not value:
            raise TransactionConflict("module %r is owned externally" % name)
        def undo():
            if modules.get(name, _MISSING) is not value:
                raise TransactionConflict("runtime hook lost module %r" % name)
            if old is _MISSING:
                modules.pop(name, None)
            else:
                modules[name] = old
        modules[name] = value
        self.record(modules, name, old, value, undo=undo)


@contextmanager
def runtime_hook(owner, parent_transaction=None):
    """Record one atomic runtime activation, preserving foreign writes on undo."""
    inherited = current_transaction()
    transaction = RuntimeHook(owner)
    transaction.acquire()
    previous = getattr(_HOOK_LOCAL, "transaction", None)
    parent = previous if previous is not None else parent_transaction
    if parent is None:
        parent = inherited
    _HOOK_LOCAL.transaction = transaction
    try:
        if parent is not None and parent.state != "open":
            raise RuntimeError("runtime hook parent must be open")
        yield transaction
    except BaseException:
        if transaction.state not in ("rolled_back", "failed"):
            transaction.rollback()
        raise
    else:
        try:
            if transaction.state == "rolled_back":
                return
            if transaction._entries and parent is not None:
                parent.record_undo(transaction.rollback)
                parent._covered_slots.update(transaction._covered_slots)
            transaction.commit()
            if transaction._entries:
                _RUNTIME_HOOKS.setdefault(owner, []).append(transaction)
        except BaseException:
            transaction.rollback()
            raise
    finally:
        _HOOK_LOCAL.transaction = previous
        transaction.release()


def owned_runtime_hook(owner):
    def decorate(function):
        @wraps(function)
        def apply(*args, **kwargs):
            with runtime_hook(owner):
                return function(*args, **kwargs)
        return apply
    return decorate


def release_runtime_hooks(owner):
    """Undo all of an owner's hooks; report conflicts after restoring others."""
    with InstallTransaction._lock:
        hooks = _RUNTIME_HOOKS.pop(owner, ())
        conflicts = []
        for hook in reversed(hooks):
            if hook.state == "rolled_back":
                continue
            try:
                hook.rollback()
            except TransactionConflict as error:
                conflicts.append(str(error))
        if conflicts:
            raise TransactionConflict("; ".join(conflicts))


def runtime_owns_module(owner, modules, name):
    with InstallTransaction._lock:
        for hook in reversed(_RUNTIME_HOOKS.get(owner, ())):
            if hook.state != "active":
                continue
            for target, key, old, new, undo, token in reversed(hook._entries):
                if target is modules and key == name:
                    return modules.get(name, _MISSING) is new
    return False


def active_transaction(context=None):
    """The transaction currently recording for ``context``, or None.

    Six installers used to inline their own copy of this lookup and five of them
    omitted the state check, so a transaction left in ``context.state`` after it
    had been committed or rolled back turned the next write into
    ``RuntimeError: transaction is <state>`` rather than a write. That matters
    beyond install: the factory and tensor owners call the same helpers from
    ``torch.zeros(device="cuda")`` at runtime, long after any ledger is closed.
    """
    hook = getattr(_HOOK_LOCAL, "transaction", None)
    if hook is not None and hook.state == "open":
        return hook
    if context is None and current_transaction() is not None:
        return current_transaction()
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
    if isinstance(target, (MutableMapping, list)):
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
    "RuntimeHook",
    "TransactionConflict",
    "active_transaction",
    "current_runtime_hook",
    "current_transaction",
    "owned_runtime_hook",
    "release_runtime_hooks",
    "runtime_hook",
    "runtime_owns_module",
    "set_attr",
    "set_env",
    "set_flag",
]
