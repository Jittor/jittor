"""Python kernel selection backed by the native runtime's device context.

Registration has no bootstrap or driver side effects. Selection examines all
tensor arguments without materializing them; selected implementations and
capability predicates propagate their errors without trying another backend.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
import sys
import threading
from typing import NamedTuple


class DispatchContext(NamedTuple):
    backend: str
    device_id: int
    dtypes: tuple


@dataclass(frozen=True)
class KernelRegistration:
    implementation: object
    dtypes: object = None
    supports: object = None
    priority: int = 0
    runtime_modes: object = None


_lock = threading.RLock()
_kernels = {}


def dispatch_context(*args, **kwargs):
    native = sys.modules.get("jittor")
    if native is None or not hasattr(native, "core"):
        raise RuntimeError("Jittor must be initialized before selecting a kernel")
    var_type = native.core.Var
    tensors = []
    active_containers = set()

    def collect(value):
        if isinstance(value, var_type):
            tensors.append(value)
        elif isinstance(value, (tuple, list, dict)):
            identity = id(value)
            if identity in active_containers:
                raise ValueError("cyclic kernel argument container")
            active_containers.add(identity)
            try:
                for child in value.values() if isinstance(value, dict) else value:
                    collect(child)
            finally:
                active_containers.remove(identity)

    collect(args)
    collect(kwargs)
    backend, device_id = native.core.dispatch_context(tensors)
    return DispatchContext(backend, device_id, tuple(str(value.dtype) for value in tensors))


def register_kernel(op, backend, implementation, *, dtypes=None,
                    supports=None, priority=0, runtime_modes=None):
    if not isinstance(op, str) or not op or not isinstance(backend, str) or not backend:
        raise ValueError("kernel operator and backend must be non-empty names")
    if not callable(implementation) or (supports is not None and not callable(supports)):
        raise TypeError("kernel implementation and predicate must be callable")
    if dtypes is not None:
        if isinstance(dtypes, str):
            raise TypeError("kernel dtypes must be a collection of dtype names")
        dtypes = frozenset(dtypes)
        if any(not isinstance(dtype, str) for dtype in dtypes):
            raise TypeError("kernel dtypes must contain dtype names")
    if isinstance(priority, bool) or not isinstance(priority, int):
        raise TypeError("kernel priority must be an integer")
    if runtime_modes is not None:
        runtime_modes = frozenset(runtime_modes)
        if any(type(mode) is not int or mode not in (0, 1, 2) for mode in runtime_modes):
            raise ValueError("kernel runtime modes must be 0, 1, or 2")
    registration = KernelRegistration(implementation, dtypes, supports, priority, runtime_modes)
    with _lock:
        key = (op, backend)
        current = _kernels.get(key, ())
        for entry in current:
            if entry.implementation is implementation:
                if entry == registration:
                    return implementation
                raise ValueError("kernel already registered with different options: %s/%s" % key)
        _kernels[key] = tuple(sorted(current + (registration,),
                                     key=lambda entry: entry.priority, reverse=True))
    return implementation


def unregister_kernel(op, backend, implementation):
    with _lock:
        key = (op, backend)
        current = _kernels.get(key, ())
        remaining = tuple(entry for entry in current if entry.implementation is not implementation)
        if len(remaining) == len(current):
            raise KeyError("kernel is not registered: %s/%s" % key)
        if remaining:
            _kernels[key] = remaining
        else:
            _kernels.pop(key)
    return implementation


def registered_kernel(op, backend):
    """Inspect publication only; this does not grant dtype/shape eligibility."""
    with _lock:
        entries = _kernels.get((op, backend), ())
        return entries[0].implementation if entries else None


def select_kernel(op, *args, **kwargs):
    context = dispatch_context(*args, **kwargs)
    with _lock:
        entries = _kernels.get((op, context.backend), ()) + _kernels.get((op, "*"), ())
    for entry in sorted(entries, key=lambda item: item.priority, reverse=True):
        if entry.runtime_modes is not None:
            if sys.modules["jittor"].runtime.use_cuda not in entry.runtime_modes:
                continue
        if entry.dtypes is not None and any(dtype not in entry.dtypes for dtype in context.dtypes):
            continue
        if entry.supports is not None and not entry.supports(*args, **kwargs):
            continue
        return entry.implementation
    return None


def try_dispatch(op, *args, **kwargs):
    implementation = select_kernel(op, *args, **kwargs)
    return None if implementation is None else implementation(*args, **kwargs)


def optional_kernel(op, backend, *, dtypes=None, supports=None, priority=0,
                    runtime_modes=None):
    backends = (backend,) if isinstance(backend, str) else tuple(backend)
    if not backends or any(not isinstance(name, str) or not name for name in backends):
        raise ValueError("kernel backends must be non-empty names")

    def decorate(implementation):
        for name in backends:
            register_kernel(op, name, implementation, dtypes=dtypes,
                            supports=supports, priority=priority, runtime_modes=runtime_modes)

        @wraps(implementation)
        def optional(*args, **kwargs):
            return try_dispatch(op, *args, **kwargs)

        return optional
    return decorate


@contextmanager
def override_kernel(op, backend, implementation, *, dtypes=None,
                    supports=None, priority=0, runtime_modes=None):
    """Temporarily replace one backend's entries, restoring absence as well."""
    with _lock:
        previous = _kernels.pop((op, backend), ())
        try:
            if implementation is not None:
                register_kernel(op, backend, implementation, dtypes=dtypes,
                                supports=supports, priority=priority, runtime_modes=runtime_modes)
        except BaseException:
            if previous:
                _kernels[(op, backend)] = previous
            raise
    try:
        yield implementation
    finally:
        with _lock:
            if previous:
                _kernels[(op, backend)] = previous
            else:
                _kernels.pop((op, backend), None)
