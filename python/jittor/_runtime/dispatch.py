"""Python kernel selection backed by the native runtime's device context.

Registration has no bootstrap or driver side effects. Selection examines all
tensor arguments without materializing them; selected implementations and
capability predicates propagate their errors without trying another backend.
"""
from jittor._core.dtypes import dtype_name as _jittor_dtype_name
from jittor._core.dtypes import is_dtype as _is_dtype

from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
import sys
import threading
from typing import Any, Callable, Dict, FrozenSet, List, NamedTuple, Optional, Tuple


class DispatchContext(NamedTuple):
    backend: str
    device_id: int
    dtypes: tuple


@dataclass(frozen=True)
class KernelRegistration:
    implementation: Callable[..., Any]
    dtypes: Optional[FrozenSet[str]] = None
    supports: Optional[Callable[..., bool]] = None
    priority: int = 0
    runtime_modes: Optional[FrozenSet[int]] = None


_lock = threading.RLock()
_kernels: Dict[Tuple[str, str], Tuple[KernelRegistration, ...]] = {}

#: Candidate lists per (op, backend), already merged with the "*" backend and
#: ordered by priority. `select_kernel` redid that merge and a `sorted` call on
#: every dispatched operator even though registration barely ever changes, and
#: a training step dispatches thousands of times. Every mutation of `_kernels`
#: replaces this map, so an entry can only be as old as the last registration.
_resolved: Dict[Tuple[str, str], Tuple[KernelRegistration, ...]] = {}


def _entry_priority(entry):
    return entry.priority


def _invalidate():
    """Drop the resolved candidate lists; callers must hold `_lock`."""
    global _resolved
    _resolved = {}


def _candidates(op, backend):
    key = (op, backend)
    entries = _resolved.get(key)
    if entries is not None:
        return entries
    with _lock:
        entries = _resolved.get(key)
        if entries is None:
            entries = _kernels.get(key, ()) + _kernels.get((op, "*"), ())
            if len(entries) > 1:
                # `sorted` is stable, so equal priorities keep the backend's
                # own entries ahead of the "*" ones, exactly as the merge did.
                entries = tuple(sorted(entries, key=_entry_priority, reverse=True))
            _resolved[key] = entries
    return entries


#: Canonical dtype names by the raw name `dtype_name` reads. That function is
#: a pure function of that one string, and it rebuilds its alias table from a
#: literal on every call; this runs once per tensor of every dispatched
#: operator, and a native NanoString reaches the `str()` arm every time.
_DTYPE_NAMES: Dict[str, str] = {}


def canonical_dtype_name(dtype):
    """`dtype_name(dtype)`, memoised on the raw name it would have read."""
    raw = getattr(dtype, "name", None)
    if raw.__class__ is not str:
        raw = getattr(dtype, "__name__", None)
        if raw.__class__ is not str:
            raw = str(dtype)
    name = _DTYPE_NAMES.get(raw)
    if name is None:
        name = _jittor_dtype_name(dtype)
        _DTYPE_NAMES[raw] = name
    return name


def _dtype_names(tensors):
    return tuple([canonical_dtype_name(value.dtype) for value in tensors])


def _canonical_backend(backend):
    return "acl" if backend == "acl_legacy" else backend


def _collect_tensors(values, var_type, tensors, active_containers):
    """Append the tensors reachable from `values` in argument order.

    One Python call per *container*, not per value.  This runs once per
    dispatched operator and is a measured 2.1 of the 3.2 us that
    `dispatch_context` used to cost (task 3.21): the old version defined a
    closure on every call and recursed into every scalar, so the flat argument
    list of a normal kernel -- five values, no nesting -- paid seven calls and
    two set mutations to find three tensors.

    `active_containers` is created by the caller only when a container is
    actually entered, so the flat case allocates no set either.  The cycle it
    guards against is a container reachable from itself; that check keeps its
    exact previous meaning, including that the same container appearing twice
    side by side is not a cycle.
    """
    for value in values:
        if isinstance(value, var_type):
            tensors.append(value)
        elif isinstance(value, (tuple, list, dict)):
            identity = id(value)
            if active_containers is None:
                active_containers = set()
            elif identity in active_containers:
                raise ValueError("cyclic kernel argument container")
            active_containers.add(identity)
            try:
                _collect_tensors(
                    value.values() if isinstance(value, dict) else value,
                    var_type, tensors, active_containers)
            finally:
                active_containers.remove(identity)


def _dispatch_backend(args, kwargs):
    """The (backend, device_id, dtypes) triple without the named-tuple box."""
    native = sys.modules.get("jittor")
    if native is None or not hasattr(native, "core"):
        raise RuntimeError("Jittor must be initialized before selecting a kernel")
    var_type = native.core.Var
    tensors: List[Any] = []
    # `args` and `kwargs` are freshly built by this call, so neither can be
    # reachable from itself and neither needs an entry in the cycle set.
    _collect_tensors(args, var_type, tensors, None)
    if kwargs:
        _collect_tensors(kwargs.values(), var_type, tensors, None)
    backend, device_id = native.core.dispatch_context(tensors)
    return _canonical_backend(backend), device_id, _dtype_names(tensors)


def dispatch_context(*args, **kwargs):
    return DispatchContext(*_dispatch_backend(args, kwargs))


def register_kernel(op, backend, implementation, *, dtypes=None,
                    supports=None, priority=0, runtime_modes=None):
    backend = _canonical_backend(backend)
    if not isinstance(op, str) or not op or not isinstance(backend, str) or not backend:
        raise ValueError("kernel operator and backend must be non-empty names")
    if not callable(implementation) or (supports is not None and not callable(supports)):
        raise TypeError("kernel implementation and predicate must be callable")
    if dtypes is not None:
        if isinstance(dtypes, str):
            raise TypeError("kernel dtypes must be a collection of dtype names")
        dtypes = frozenset(_jittor_dtype_name(value) if _is_dtype(value) else value
                           for value in dtypes)
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
                                     key=_entry_priority, reverse=True))
        _invalidate()
    return implementation


def unregister_kernel(op, backend, implementation):
    backend = _canonical_backend(backend)
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
        _invalidate()
    return implementation


def registered_kernel(op, backend):
    """Inspect publication only; this does not grant dtype/shape eligibility."""
    backend = _canonical_backend(backend)
    with _lock:
        entries = _kernels.get((op, backend), ())
        return entries[0].implementation if entries else None


def select_kernel(op, *args, **kwargs):
    # Through the module-level name, not `_dispatch_backend`: replacing
    # `dispatch_context` is how a caller states which device the arguments are
    # on, and the ACL clamp facade's CPU contract is tested that way.
    context = dispatch_context(*args, **kwargs)
    backend, dtypes = context.backend, context.dtypes
    for entry in _candidates(op, backend):
        if entry.runtime_modes is not None:
            runtime_mode = sys.modules["jittor"].runtime.use_cuda
            # Explicit CPU/CUDA tensor placement can differ from the Runtime
            # default; mode eligibility follows the selected graph backend.
            selected_mode = 0 if backend == "cpu" else (runtime_mode or 1)
            if selected_mode not in entry.runtime_modes:
                continue
        if entry.dtypes is not None:
            entry_dtypes = entry.dtypes
            if any(dtype not in entry_dtypes for dtype in dtypes):
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
    backend = _canonical_backend(backend)
    with _lock:
        previous = _kernels.pop((op, backend), ())
        _invalidate()
        try:
            if implementation is not None:
                register_kernel(op, backend, implementation, dtypes=dtypes,
                                supports=supports, priority=priority, runtime_modes=runtime_modes)
        except BaseException:
            if previous:
                _kernels[(op, backend)] = previous
            _invalidate()
            raise
    try:
        yield implementation
    finally:
        with _lock:
            if previous:
                _kernels[(op, backend)] = previous
            else:
                _kernels.pop((op, backend), None)
            _invalidate()
