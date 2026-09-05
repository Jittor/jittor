"""Private state shared by the Torch tensor compatibility installers.

The compatibility package historically attached its leaf and ``retain_grad``
registries directly to the public :mod:`jittor` module.  Keeping those maps in
one state object makes their ownership explicit while preserving the existing
``torch is jittor`` publication contract.
"""

from __future__ import annotations

class TorchTensorState(dict):
    """Per-installed-module bookkeeping for Torch-facing tensor autograd.

    The object subclasses ``dict`` so the historical ``jt._torch_leaf_params``
    mapping remains source-compatible while the retained registry is grouped
    beside it.  This also keeps the published module namespace unchanged.
    """

    def __init__(self):
        super().__init__()
        self.retained = {}
        # Torch's requires_grad state is owned here as well.  Jittor Vars are
        # not weak-referenceable, so retain the object only while the
        # torch-facing flag is enabled; the backward bridge prunes stale leaf
        # entries as before.
        self.requires_grad = {}
        # Weak references to every live Torch-compatible optimizer.  Keep this
        # beside leaf/retain state so installers share one ownership boundary.
        self.active_optimizers = []

    @property
    def leaf_params(self):
        return self

    def set_requires_grad(self, tensor, enabled):
        """Record the torch-facing requires-grad bit and return ``enabled``."""
        key = id(tensor)
        if enabled:
            self.requires_grad[key] = tensor
        else:
            self.requires_grad.pop(key, None)
        return bool(enabled)

    def clear_requires_grad(self, tensor):
        """Forget a tensor whose native gradient flag was stopped in-place."""
        self.requires_grad.pop(id(tensor), None)
        return tensor

    def requires_grad_tensors(self):
        """Return a snapshot of tensors explicitly enabled for gradients."""
        return tuple(self.requires_grad.values())


def get_tensor_state(jittor_module):
    """Return the state owned by *jittor_module*, creating it once.

    The state is deliberately attached to the published module rather than to
    the compatibility package: repeated installs and legacy imports therefore
    observe the same state object without changing module identity.
    """
    # The explicit owner is the migration seam for an eventual independent
    # torch package.  Legacy aliases remain published below for callers that
    # still inspect the old private names.
    state = getattr(jittor_module, "_torch_tensor_state", None)
    if not isinstance(state, TorchTensorState):
        state = getattr(jittor_module, "_torch_leaf_params", None)
    legacy_retained = getattr(jittor_module, "_torch_retained", None)
    if not isinstance(state, TorchTensorState):
        previous = state if isinstance(state, dict) else None
        state = TorchTensorState()
        if previous:
            state.update(previous)
        # A process can be upgraded after an older install has already
        # registered retained tensors. Preserve those entries while moving
        # the registry behind the state owner.
        if isinstance(legacy_retained, dict):
            state.retained.update(legacy_retained)
    setattr(jittor_module, "_torch_tensor_state", state)
    if getattr(jittor_module, "_torch_leaf_params", None) is not state:
        setattr(jittor_module, "_torch_leaf_params", state)
    # Older installs published the optimizer registry independently.  Adopt it
    # once when upgrading an existing process, then keep the old name as an
    # identity-preserving compatibility alias.
    legacy_optimizers = getattr(jittor_module, "_active_optimizers", None)
    if isinstance(legacy_optimizers, list) and not state.active_optimizers:
        state.active_optimizers = legacy_optimizers
    if getattr(jittor_module, "_active_optimizers", None) is not state.active_optimizers:
        setattr(jittor_module, "_active_optimizers", state.active_optimizers)
    if getattr(jittor_module, "_torch_retained", None) is not state.retained:
        setattr(jittor_module, "_torch_retained", state.retained)
    return state


__all__ = ["TorchTensorState", "get_tensor_state"]
