"""Opt-in empty Linear construction for checkpoint loaders."""

from contextvars import ContextVar


_DEFER_LINEAR_INITIALIZATION = ContextVar("_DEFER_LINEAR_INITIALIZATION", default=False)


def load_with_deferred_linear_initialization(loader, *args, **kwargs):
    """Run a loader whose constructors only inspect parameter metadata.

    The loader must install each Linear's checkpoint storage with
    ``load_state_dict(assign=True)`` and initialize parameters missing from the
    checkpoint. Return the model, optionally followed by its loading report.
    Only opt in for loaders validated against those requirements.
    """
    token = _DEFER_LINEAR_INITIALIZATION.set(True)
    try:
        result = loader(*args, **kwargs)
    finally:
        _DEFER_LINEAR_INITIALIZATION.reset(token)
    model = result[0] if isinstance(result, tuple) else result
    pending = [name for name, param in model.named_parameters()
               if getattr(param, "_jittor_checkpoint_linear_meta", False)]
    if pending:
        raise RuntimeError(f"Checkpoint loader left Linear parameters on meta: {', '.join(pending)}")
    return result


def deferred_linear_initialization_active():
    return _DEFER_LINEAR_INITIALIZATION.get()


def make_deferred_linear_parameter(torch, shape, dtype):
    data = torch.empty(shape, device="meta", dtype=dtype)
    param = torch.nn.Parameter(data)
    # Parameter construction detaches data into a new holder. Give this
    # holder the meta mark used by loaders when materializing missing keys.
    param._jittor_torch_meta = True
    param._jittor_checkpoint_linear_meta = True
    return param


def finish_deferred_linear_parameter(target):
    if not getattr(target, "_jittor_checkpoint_linear_meta", False):
        return
    # Native assign updates the original Var holder, so its Python marks
    # survive replacement. Clear them only after the successful assignment.
    del target._jittor_torch_meta
    del target._jittor_checkpoint_linear_meta
