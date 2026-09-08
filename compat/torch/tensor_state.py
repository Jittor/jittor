"""Private state shared by the Torch tensor compatibility installers.

The compatibility package historically attached its leaf and ``retain_grad``
registries directly to the public :mod:`jittor` module.  Keeping those maps in
one state object makes their ownership explicit. A native backend resolves its
explicit active compatibility owner through a private weak module binding.
Independent installation publishes state only on its target namespace;
historical native attributes are supported only by legacy initialization.
"""

from __future__ import annotations

from types import ModuleType
from weakref import WeakKeyDictionary, ref

from ..transaction import TransactionConflict, _MISSING
from .holder_registry import HolderRegistry


_OWNERS = WeakKeyDictionary()


def _bound_owner(module):
    binding = _OWNERS.get(module) if isinstance(module, ModuleType) else None
    if binding is None:
        return module
    owner = binding()
    if owner is None:
        _OWNERS.pop(module, None)
        return module
    return owner

class TorchTensorState(HolderRegistry):
    """Per-installed-module bookkeeping for Torch-facing tensor autograd.

    The object subclasses ``dict`` so the historical ``jt._torch_leaf_params``
    mapping remains source-compatible while the retained registry is grouped
    beside it. Independent holders are weak entries; the dictionaries locate
    Python objects and never define their leaf identity or extend their life.
    """

    def __init__(self):
        super().__init__()
        self.retained = HolderRegistry()
        # The native tensor descriptor owns requires_grad. Only registries
        # consumed by backward belong here; do not retain tensors a second
        # time merely because their gradient flag was enabled.
        # Weak references to every live Torch-compatible optimizer.  Keep this
        # beside leaf/retain state so installers share one ownership boundary.
        self.active_optimizers = []

    @property
    def leaf_params(self):
        return self

def compatibility_owner(module):
    """Resolve an explicitly bound owner, never an inferred sys.modules root."""
    owner = _bound_owner(module)
    if owner is None or not hasattr(owner, "__dict__"):
        raise RuntimeError("invalid Torch compatibility owner binding")
    if _bound_owner(owner) is not owner:
        raise RuntimeError("Torch compatibility owner bindings must not form chains")
    return owner


def _existing_state(module):
    local = vars(module)
    state = local.get("_torch_tensor_state")
    if not isinstance(state, TorchTensorState):
        state = local.get("_torch_leaf_params")
    return state if isinstance(state, TorchTensorState) else None


def _adopt_legacy_state(module):
    local = vars(module)
    state = TorchTensorState()
    leaves = local.get("_torch_leaf_params")
    if isinstance(leaves, dict):
        state.update(leaves)
    retained = local.get("_torch_retained")
    if isinstance(retained, dict):
        state.retained.update(retained)
    optimizers = local.get("_active_optimizers")
    if isinstance(optimizers, list):
        state.active_optimizers = optimizers
    return state


def _publish_state(module, state, transaction=None):
    aliases = {
        "_torch_tensor_state": state,
        "_torch_leaf_params": state,
        "_torch_retained": state.retained,
        "_active_optimizers": state.active_optimizers,
    }
    for name, value in aliases.items():
        if vars(module).get(name, _MISSING) is value:
            continue
        if transaction is None:
            setattr(module, name, value)
        else:
            transaction.mutate_attr(module, name, value)


def get_tensor_state(jittor_module):
    """Return one active-owner state, retaining unbound legacy initialization."""
    owner = compatibility_owner(jittor_module)
    bound = isinstance(owner, ModuleType) and owner in _OWNERS
    state = vars(owner).get("_torch_tensor_state") if bound else _existing_state(owner)
    if not isinstance(state, TorchTensorState):
        if bound:
            raise RuntimeError("bound Torch owner has no local tensor state")
        state = _adopt_legacy_state(owner)
    if not bound:
        optimizers = vars(owner).get("_active_optimizers")
        if isinstance(optimizers, list) and not state.active_optimizers:
            state.active_optimizers = optimizers
    _publish_state(owner, state)
    return state


def latest_optimizer(module):
    """Resolve the most recently registered live optimizer without owning it."""
    state = _existing_state(compatibility_owner(module))
    if state is None:
        return None
    registry = state.active_optimizers
    for reference in reversed(registry):
        optimizer = reference()
        if optimizer is not None:
            return optimizer
    return None


def bind_tensor_state(native_backend, target, transaction, state=None):
    """Provision one owner before installers run; every binding is reversible."""
    if transaction.state != "open":
        raise RuntimeError("tensor-state binding requires an open install transaction")
    if not isinstance(native_backend, ModuleType) or not isinstance(target, ModuleType):
        raise TypeError("Torch owner binding requires module objects")
    previous_owner = _bound_owner(native_backend)
    if previous_owner is not native_backend and previous_owner is not target:
        raise RuntimeError("native backend already has a different active Torch owner")
    if _bound_owner(target) is not target:
        raise RuntimeError("target is already bound to a different Torch owner")
    for owner in (native_backend, target):
        if owner in _OWNERS:
            bound_owner = compatibility_owner(owner)
            if not isinstance(vars(bound_owner).get("_torch_tensor_state"), TorchTensorState):
                raise RuntimeError("bound Torch owner has no local tensor state")
    if state is not None and not isinstance(state, TorchTensorState):
        raise TypeError("tensor state must be a TorchTensorState")
    candidates = [item for item in
                  (state, _existing_state(native_backend), _existing_state(target))
                  if item is not None]
    if candidates and any(item is not candidates[0] for item in candidates[1:]):
        raise RuntimeError("native backend and target have conflicting tensor states")
    state = candidates[0] if candidates else None
    if state is not None:
        for owner in (native_backend, target):
            if _existing_state(owner) is None and any(
                    vars(owner).get(name) for name in
                    ("_torch_leaf_params", "_torch_retained", "_active_optimizers")):
                raise RuntimeError("existing tensor state conflicts with legacy registries")
    if state is None:
        sources = []
        for owner in (native_backend, target):
            if any(vars(owner).get(name) for name in
                   ("_torch_leaf_params", "_torch_retained", "_active_optimizers")):
                if all(owner is not previous for previous in sources):
                    sources.append(owner)
        if len(sources) > 1:
            raise RuntimeError("native backend and target have conflicting legacy registries")
        # Preserve an existing empty optimizer list's identity as well.
        state = _adopt_legacy_state(sources[0] if sources else native_backend)
    _publish_state(target, state, transaction)
    for owner in (target,) if target is native_backend else (target, native_backend):
        previous = _OWNERS.get(owner, _MISSING)
        if previous is _MISSING or previous() is not target:
            binding = ref(target)
            transaction.record(_OWNERS, owner, previous, binding)
            _OWNERS[owner] = binding
    return state


def snapshot_tensor_state(state):
    """Capture entry identities, preserving the three original containers."""
    return {
        "state": state,
        "leaves": dict(state),
        "retained": state.retained,
        "retained_entries": dict(state.retained),
        "optimizers": state.active_optimizers,
        "optimizer_entries": tuple(state.active_optimizers),
    }


def _record_mapping_changes(transaction, mapping, before):
    transaction.record_mapping_diffs(mapping, before)


def _record_list_entry(transaction, values, index, old, new):
    def undo():
        current = values[index] if index < len(values) else _MISSING
        if current is not new:
            raise TransactionConflict("optimizer registry entry %r changed externally" % index)
        if old is _MISSING:
            del values[index]
        elif new is _MISSING:
            if len(values) != index:
                raise TransactionConflict("optimizer registry insertion boundary changed")
            values.insert(index, old)
        else:
            values[index] = old
    transaction.record(values, index, old, new, undo=undo)


def record_tensor_state_changes(transaction, state, before):
    """Record per-entry undo before rollback; never replace a whole registry."""
    if before["state"] is not state:
        raise ValueError("tensor-state snapshot belongs to a different state")
    _record_mapping_changes(transaction, state, before["leaves"])
    retained = before["retained"]
    _record_mapping_changes(transaction, retained, before["retained_entries"])
    values, old = before["optimizers"], before["optimizer_entries"]
    # Removed tail entries undo in ascending order, after existing slots have
    # been restored. Appended slots undo from the end and preserve foreign tails.
    for index in reversed(range(len(values), len(old))):
        _record_list_entry(transaction, values, index, old[index], _MISSING)
    for index in range(min(len(old), len(values))):
        if old[index] is not values[index]:
            _record_list_entry(transaction, values, index, old[index], values[index])
    for index in range(len(old), len(values)):
        _record_list_entry(transaction, values, index, _MISSING, values[index])
    for name, original in (("retained", retained), ("active_optimizers", values)):
        current = getattr(state, name)
        if current is not original:
            transaction.record(state, name, original, current,
                               undo=_attribute_undo(state, name, original, current))


def _attribute_undo(state, name, original, expected):
    # TorchTensorState is a dict subclass, so the generic ledger's dict entry
    # handling cannot restore these two object attributes.
    def undo():
        if getattr(state, name) is not expected:
            raise TransactionConflict("tensor-state container %r changed externally" % name)
        setattr(state, name, original)
    return undo


__all__ = [
    "TorchTensorState", "get_tensor_state", "compatibility_owner",
    "bind_tensor_state", "snapshot_tensor_state", "record_tensor_state_changes",
]
