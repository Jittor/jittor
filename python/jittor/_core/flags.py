"""Native flag scopes and their single shared runtime state."""

import functools as _functools

import jittor_core as core
from jittor_core import Var, sync_all

from jittor._runtime.state import RuntimeContext, RuntimeState


class _call_no_record_scope:
    def __enter__(self): pass
    def __exit__(self, *exc): pass

    def _new_scope(self):
        """A fresh scope object for one decorated call.

        ``__call__`` used to close over ``self`` and re-enter that one instance
        on every call, which is half of why ``@jt.no_grad()`` leaked: a
        recursive call re-entered the same object. Scopes that keep no
        per-entry state can go on returning ``self``; :class:`flag_scope`
        overrides this.
        """
        return self

    def __call__(self, func):
        @_functools.wraps(func)
        def inner(*args, **kw):
            with self._new_scope():
                ret = func(*args, **kw)
            return ret
        return inner

class flag_scope(_call_no_record_scope):
    """Set jittor flags for the duration of a ``with`` block or a call.

    The saved values live on a **stack**, not on a single attribute. With one
    attribute, entering the same scope object twice without leaving it in
    between overwrote the outer entry's backup with the inner one's, and the
    outer ``__exit__`` then "restored" the inner scope's values -- permanently.

    That is not an exotic case: ``__call__`` decorates a function with one
    scope instance, so ``@jt.no_grad()`` on a **recursive** function hits it on
    the first recursive call. The outer frame saved ``no_grad=0``, the inner
    frame overwrote that backup with ``no_grad=1`` (already inside the scope),
    and on the way out the process was left with ``no_grad=1`` for good --
    every subsequent ``jt.grad`` silently returning zeros, no error, training
    loss simply not moving.
    """

    def __init__(self, **jt_flags):
        self.jt_flags = jt_flags
        # one entry per active __enter__, so nesting and recursion compose
        self._flags_bk_stack = []

    def _new_scope(self):
        # a decorated call gets its own scope object as well as its own stack
        # entry; both are needed for reentrancy across threads and generators
        return type(self)(**self.jt_flags)

    def _flush_if_device_changes(self, wanted):
        """Run everything still pending before the device flag moves.

        Execution is lazy, so an op built inside ``flag_scope(use_cuda=1)`` can
        still be pending when the scope restores the old value, and it then runs
        on the other device -- with an op that has no CPU version (a bfloat16
        cast, say) that is a hard failure, and with one that has both it is a
        silent device switch. Only a scope that actually moves ``use_cuda`` pays
        for this, and such a scope is a device boundary anyway.
        """
        # Compare device mode without constructing scalar tensor operations.
        if "use_cuda" not in self.jt_flags:
            return
        current = getattr(flags, "use_cuda")
        if (current != 0) == (wanted != 0):
            return
        sync_all()

    def __enter__(self):
        flags_bk = {}
        # push BEFORE setting anything, so the __exit__ in the except branch
        # below pops this entry and not an enclosing scope's
        self._flags_bk_stack.append(flags_bk)
        try:
            if "use_cuda" in self.jt_flags:
                self._flush_if_device_changes(self.jt_flags["use_cuda"])
            for k,v in self.jt_flags.items():
                origin = getattr(flags, k)
                flags_bk[k] = origin
                # merge dict attrs
                if isinstance(origin, dict):
                    for ok, ov in origin.items():
                        if ok not in v:
                            v[ok] = ov
                setattr(flags, k, v)
        except:
            self.__exit__()
            raise

    def __exit__(self, *exc):
        if not self._flags_bk_stack:
            # __exit__ without a matching __enter__; nothing was saved
            return
        flags_bk = self._flags_bk_stack.pop()
        # Not while an exception is unwinding: the pending work is likely what
        # raised, and a second error here would bury the first one.
        unwinding = len(exc) > 0 and exc[0] is not None
        try:
            if "use_cuda" in flags_bk and not unwinding:
                self._flush_if_device_changes(flags_bk["use_cuda"])
        finally:
            # Restoring the flags is not optional: leaving the scope's values in
            # place because the flush raised would corrupt everything after it.
            for k,v in flags_bk.items():
                setattr(flags, k, v)

class no_grad(flag_scope):
    ''' no_grad scope, all variable created inside this
scope will stop grad.

Example::

    import jittor as jt

    with jt.no_grad():
        ...

    '''
    def __init__(self, **jt_flags):
        jt_flags["no_grad"] = 1
        super().__init__(**jt_flags)

class enable_grad(flag_scope):
    ''' enable_grad scope, all variable created inside this
scope will start grad.

Example::

    import jittor as jt

    with jt.enable_grad():
        ...

    '''
    def __init__(self, **jt_flags):
        jt_flags["no_grad"] = 0
        super().__init__(**jt_flags)


def _output_requires_grad(*values):
    """Whether an op built from ``values`` must preserve an autograd path.

    Process-wide grad mode is only half of that decision.  Outside a
    ``no_grad`` scope an operation whose tensor inputs are all stopped still
    produces a stopped output, so inference-only fused kernels are safe for
    it.  Containers are accepted because stack/cache dispatchers receive
    lists and dictionaries of tensors.
    """
    if flags.no_grad:
        return False
    pending = list(values)
    while pending:
        value = pending.pop()
        if isinstance(value, Var):
            if value.requires_grad:
                return True
        elif isinstance(value, (list, tuple)):
            pending.extend(value)
        elif isinstance(value, dict):
            pending.extend(value.values())
    return False


def _stop_grad_outputs(value):
    """Mark an inference fusion's returned tensors as non-differentiable."""
    if isinstance(value, Var):
        value.stop_grad()
    elif isinstance(value, list):
        for item in value:
            _stop_grad_outputs(item)
    elif isinstance(value, tuple):
        for item in value:
            _stop_grad_outputs(item)
    elif isinstance(value, dict):
        for item in value.values():
            _stop_grad_outputs(item)
    return value

_core_flags = core.Flags()
flags = _core_flags


_runtime_context = RuntimeContext(flags)
runtime = RuntimeState(_runtime_context, flag_scope)
