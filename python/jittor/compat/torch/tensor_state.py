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

    @property
    def leaf_params(self):
        return self


def get_tensor_state(jittor_module):
    """Return the state owned by *jittor_module*, creating it once.

    The state is deliberately attached to the published module rather than to
    the compatibility package: repeated installs and legacy imports therefore
    observe the same state object without changing module identity.
    """
    state = getattr(jittor_module, "_torch_leaf_params", None)
    if not isinstance(state, TorchTensorState):
        previous = state if isinstance(state, dict) else None
        state = TorchTensorState()
        if previous:
            state.update(previous)
        setattr(jittor_module, "_torch_leaf_params", state)
    if getattr(jittor_module, "_torch_retained", None) is not state.retained:
        setattr(jittor_module, "_torch_retained", state.retained)
    return state


__all__ = ["TorchTensorState", "get_tensor_state"]
