"""Owned metadata for the existing storage-backed meta compatibility contract.

These placeholders carry ordinary native storage. They support HF's shape/key
inspection and materialization flow, never allocation-free construction or
memory-capacity offload. A marker describes its own holder, not every tensor
read while a meta context is active.
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
    """Publish constructor metadata using the same precedence as placement."""
    return mark_placeholder(value, requested(device, like))
