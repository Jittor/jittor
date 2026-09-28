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

#: Every op name that has a registration for any backend. An op absent from
#: this set cannot match on any backend, so `select_kernel` can answer None
#: without walking the arguments or asking the runtime where they live. That
#: walk is most of a dispatch's cost, and the misses are not rare: with cuTT
#: absent, every `tensor.transpose` in every model is one.
_registered_ops: FrozenSet[str] = frozenset()


def _rebuild_op_names():
    """Callers must hold `_lock`."""
    global _registered_ops
    _registered_ops = frozenset(op for op, _backend in _kernels)


def _entry_priority(entry):
    return entry.priority


def _invalidate():
    """Drop the resolved candidate lists; callers must hold `_lock`."""
    global _resolved
    _resolved = {}
    _rebuild_op_names()


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
    """Canonical dtype names of Vars, without a call per tensor.

    A Var's dtype is always a native ``NanoString``, which carries neither
    ``name`` nor ``__name__``; ``canonical_dtype_name`` therefore reaches
    ``str()`` for every one of them after two failed attribute lookups. The
    memo is the same one, so an unusual spelling still resolves through
    ``dtype_name`` exactly once.
    """
    if _VAR_TYPE is None:
        _bind_core()
    native_dtype = _NATIVE_DTYPE
    names = []
    for value in tensors:
        dtype = value.dtype if native_dtype is None else native_dtype.__get__(value, None)
        raw = str(dtype)
        name = _DTYPE_NAMES.get(raw)
        if name is None:
            name = _jittor_dtype_name(dtype)
            _DTYPE_NAMES[raw] = name
        names.append(name)
    return tuple(names)


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


#: `jittor.core` and `jittor.core.Var`, looked up once. They are the same
#: objects for the life of the process once the extension is imported, and
#: finding them cost a `sys.modules` read, a `hasattr`, and two attribute
#: lookups on every dispatched operator -- of which a decode step has a
#: hundred.
_VAR_TYPE = None
_DISPATCH_CONTEXT_NATIVE = None
#: The native Var's own `dtype` getter. A frontend tensor type overrides
#: `dtype` with a Python property returning its framework's dtype object; the
#: dispatcher only needs the name, and reading it here skips that property --
#: a Python call per tensor argument of every dispatched operator.
_NATIVE_DTYPE = None


def _bind_core():
    """Bind the native handles, or say that Jittor is not up yet."""
    global _VAR_TYPE, _DISPATCH_CONTEXT_NATIVE, _NATIVE_DTYPE
    native = sys.modules.get("jittor")
    if native is None or not hasattr(native, "core"):
        raise RuntimeError("Jittor must be initialized before selecting a kernel")
    core = native.core
    _VAR_TYPE = core.Var
    _DISPATCH_CONTEXT_NATIVE = core.dispatch_context
    _NATIVE_DTYPE = core.Var.__dict__.get("dtype")


def _walk_container(container, var_type, tensors):
    """One container argument, cheaply when it holds no container of its own.

    A container argument is almost always a shape, a permutation or a dim list
    -- a flat tuple of ints. `_collect_tensors` handles the general case, but
    to do so it allocates a cycle set, an id, an add, a `try`/`finally` and a
    remove for every container it enters, including these. That bookkeeping
    only means anything once a container is reachable from inside itself, which
    needs at least one nested container, so it is deferred until one is seen --
    at which point whatever this found is dropped and the general walk redoes
    the whole container, cycle set and all.
    """
    mark = len(tensors)
    values = container.values() if isinstance(container, dict) else container
    for value in values:
        if isinstance(value, var_type):
            tensors.append(value)
        elif isinstance(value, (tuple, list, dict)):
            del tensors[mark:]
            _collect_tensors((container,), var_type, tensors, None)
            return


def _dispatch_placement(args, kwargs):
    """The argument Vars and the (backend, device_id) the runtime puts them on."""
    var_type = _VAR_TYPE
    if var_type is None:
        _bind_core()
        var_type = _VAR_TYPE
    tensors: List[Any] = []
    # `args` and `kwargs` are freshly built by this call, so neither can be
    # reachable from itself and neither needs an entry in the cycle set. A
    # kernel's arguments are almost always flat, so that case is walked here
    # and `_collect_tensors` is entered only for a container that really has
    # to be descended into.
    for value in args:
        if isinstance(value, var_type):
            tensors.append(value)
        elif isinstance(value, (tuple, list, dict)):
            _walk_container(value, var_type, tensors)
    if kwargs:
        for value in kwargs.values():
            if isinstance(value, var_type):
                tensors.append(value)
            elif isinstance(value, (tuple, list, dict)):
                _walk_container(value, var_type, tensors)
    backend, device_id = _DISPATCH_CONTEXT_NATIVE(tensors)
    # `_canonical_backend` inlined: it is one comparison, and this is the
    # innermost frame of every dispatched operator.
    return tensors, ("acl" if backend == "acl_legacy" else backend), device_id


def _dispatch_backend(args, kwargs):
    """The (backend, device_id, dtypes) triple without the named-tuple box."""
    tensors, backend, device_id = _dispatch_placement(args, kwargs)
    return backend, device_id, _dtype_names(tensors)


def dispatch_context(*args, **kwargs):
    return DispatchContext(*_dispatch_backend(args, kwargs))


#: The function `select_kernel` reads placement through unless a caller has
#: replaced the module attribute. Kept so the fast path can tell "nobody
#: overrode this" from "a test states the device itself"; the override is a
#: published contract, so it cannot simply be bypassed.
_NATIVE_DISPATCH_CONTEXT = dispatch_context


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
    # on, and the ACL clamp facade's CPU contract is tested that way. Only when
    # nobody has replaced it does this read placement directly, which lets the
    # dtype names wait until a candidate actually filters on them. Not one ACL
    # registration declares `dtypes`, and the CUDA softmax entry that does is
    # only a candidate on a CUDA backend, so on ACL the names were built --
    # a `str` and a memo lookup per argument -- and then never read.
    # Nothing is registered under this name on any backend, so no argument
    # walk and no placement query can change the answer.
    if op not in _registered_ops:
        return None
    if dispatch_context is _NATIVE_DISPATCH_CONTEXT:
        tensors, backend, _device_id = _dispatch_placement(args, kwargs)
        dtypes = None
    else:
        context = dispatch_context(*args, **kwargs)
        tensors, backend, dtypes = None, context.backend, context.dtypes
    # `_candidates` resolves and memoizes; once it has, the answer is a plain
    # dict read, so take it here rather than through another frame.
    entries = _resolved.get((op, backend))
    if entries is None:
        entries = _candidates(op, backend)
    for entry in entries:
        if entry.runtime_modes is not None:
            runtime_mode = sys.modules["jittor"].runtime.use_cuda
            # Explicit CPU/CUDA tensor placement can differ from the Runtime
            # default; mode eligibility follows the selected graph backend.
            selected_mode = 0 if backend == "cpu" else (runtime_mode or 1)
            if selected_mode not in entry.runtime_modes:
                continue
        if entry.dtypes is not None:
            if dtypes is None and tensors is not None:
                dtypes = _dtype_names(tensors)
            # A set test, not a generator: `entry.dtypes` is a frozenset and
            # this runs per candidate of every dtype-filtered operator.
            if not entry.dtypes.issuperset(dtypes):
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
            # `try_dispatch` inlined. This wrapper is the published entry point
            # of every optional kernel -- `nn.gelu`, `nn.softmax`,
            # `nn.layer_norm.training`, `tensor.transpose` -- so the extra
            # frame is on the hot path of every one of them.
            selected = select_kernel(op, *args, **kwargs)
            return None if selected is None else selected(*args, **kwargs)

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
