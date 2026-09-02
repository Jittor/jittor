"""The seam that lets the tensor and optimizer layers behave correctly for
FSDP-managed parameters without importing FSDP.

The intended layering is ``core -> tensor -> nn/optim -> distributed -> fsdp``.
What was there instead was a two-way edge: ``installers/tensor.py``,
``optimizers.py`` and ``installers/nn.py`` each reached *up* with an inline
``from jittor.compat import fsdp2`` in the middle of a hot path, while
``fsdp2/installer.py`` reached back *down* into ``compat.torch.context``. Two
packages that import each other cannot be read, tested or replaced apart, and
the inline imports additionally put the whole FSDP2 package behind a per-call
``try``/``except``, so a genuine error raised while importing it was
indistinguishable from "FSDP is not installed here".

This module is the inversion. It belongs to neither side -- it is a leaf with
no imports at all -- so both may depend on it:

* ``jittor.compat.fsdp2`` calls :func:`register` when it is imported;
* the lower layers call :func:`provider` and take their non-FSDP path when it
  returns ``None``.

**Why ``None`` is always safe.** Every one of these call sites first tests a
marker on the objects themselves -- ``_jittor_fsdp2_state`` on a parameter,
``Module._fsdp_state``, ``_is_fsdp_module``. Only fsdp2 ever sets those
markers, so a process holding one has necessarily imported
``jittor.compat.fsdp2``, which has necessarily run :func:`register`.
``provider() is None`` therefore means exactly what the old ``except
ImportError`` path meant: there is no FSDP in this process, take the ordinary
route. The difference is that it is now a fact the code can state, rather than
an exception it had to provoke to find out.

Registration is validated against :data:`REQUIRED` so that an fsdp2 which stops
exporting one of these names fails loudly at import, instead of as an
``AttributeError`` thrown from the middle of somebody's training step.
"""

from __future__ import absolute_import

__all__ = ["register", "provider", "is_registered", "reset", "REQUIRED"]


#: The names the layers below fsdp2 call on the provider. This tuple *is* the
#: contract between the two packages: ``jittor.compat.fsdp2`` supplies all of
#: them, and nothing below it may call anything that is not named here.
#:
#: The underscore-prefixed ones are fsdp2's own spelling. They are not renamed
#: on the way out because fsdp2's *public* surface is the torch-facing API and
#: is pinned byte-for-byte by tests/structure/test_torch_fsdp2_structure.py;
#: adding seam plumbing to it would widen the torch API for no one's benefit.
REQUIRED = (
    # installers/tensor.py, from Tensor.backward(): which params FSDP owns,
    # which full params the backward must treat as leaves, and how the
    # resulting grad map is handed back to the sharded optimizers.
    "is_fsdp_managed_param",
    "collect_fsdp_full_params_for_backward",
    "fill_fsdp_optimizer_grads_from_grad_map",
    # optimizers.py, from step() and zero_grad().
    "optimizer_has_fsdp_params",
    "optimizer_has_non_fsdp_params",
    "optimizer_step",
    "clear_fsdp_optimizer_grads",
    "refresh_visible_full_grads",
    # installers/nn.py, from Module.__call__: run one forward with this
    # module's parameters unsharded.
    "_execute_with_true_fsdp",
    # installers/distributed.py, from the distributed-checkpoint state dict.
    "_load_full_state_dict",
)

_provider = None


def register(implementation):
    """Publish the FSDP implementation to the layers below it.

    Called by ``jittor.compat.fsdp2`` at import time -- deliberately not at
    install time, because a parameter can carry an FSDP marker in a process
    where the torch shim's installers never ran.
    """
    missing = [name for name in REQUIRED
               if not callable(getattr(implementation, name, None))]
    if missing:
        raise TypeError(
            "%r cannot serve as the FSDP provider: missing %s. The layers "
            "below fsdp2 call exactly the names in fsdp_hooks.REQUIRED; if one "
            "of them was renamed, both sides have to move together."
            % (implementation, ", ".join(missing)))
    global _provider
    _provider = implementation
    return implementation


def provider():
    """The registered FSDP implementation, or ``None`` if fsdp2 is not loaded.

    ``None`` is an ordinary answer, not an error: see the module docstring for
    why no process that owns an FSDP-managed parameter can observe it.
    """
    return _provider


def is_registered():
    """Whether ``jittor.compat.fsdp2`` has been imported in this process."""
    return _provider is not None


def reset():
    """Drop the registration and return it (tests only)."""
    global _provider
    previous = _provider
    _provider = None
    return previous
