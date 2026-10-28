"""Whether a Torch autocast region is active on this thread.

The amp register is shared: ``torch.autocast`` and native
``auto_mixed_precision_level`` both describe themselves by setting
``jt.flags.amp_reg``.  Operators that ATen keeps on its float32 autocast list
need the distinction -- ``layer_norm`` returns float32 inside an autocast
region, while native AMP keeps the half input's public dtype.  The Torch
frontend records its region here, per thread, and native code reads it.
"""

import threading

_state = threading.local()


def set_torch_autocast_active(active):
    """Record whether the calling thread is inside a Torch autocast region."""
    _state.active = bool(active)


def torch_autocast_active():
    """Whether the calling thread is inside a Torch autocast region."""
    return bool(getattr(_state, "active", False))


__all__ = ["set_torch_autocast_active", "torch_autocast_active"]
