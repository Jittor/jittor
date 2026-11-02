"""Metadata for storage-backed Torch meta placeholders.

Jittor does not provide allocation-free meta tensors. These placeholders use
ordinary native storage and carry per-Var metadata for shape and loader flows;
they do not provide memory-capacity offload.
"""

from .tensor_object_state import get_tensor_object_state


def is_placeholder(value):
    return bool(getattr(value, "_jittor_torch_meta", False))


def mark_placeholder(value, enabled=True):
    import jittor as backend

    if isinstance(value, backend.Var):
        state = get_tensor_object_state(value)
        if enabled or state is not None or "_jittor_torch_meta" in vars(value):
            value._jittor_torch_meta = bool(enabled)
    return value


def require_materialized(value):
    if is_placeholder(value):
        raise NotImplementedError("Cannot copy out of meta tensor; no data!")


def requested(device=None, like=None):
    if device is None:
        import jittor as backend

        if isinstance(like, backend.Var):
            return is_placeholder(like)
        from .types import active_device_context
        from .frontend import default_device

        device = active_device_context() or default_device()
    return (getattr(device, "type", None) or str(device).split(":", 1)[0]) == "meta"


def finish(value, device=None, like=None):
    """Apply the requested placeholder identity to one constructor result."""
    return mark_placeholder(value, requested(device, like))
