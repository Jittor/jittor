"""Torch numerical factories operations."""

def randint_like(input, low, high=None, dtype=None, device=None,
                 requires_grad=False, **kwargs):
    """Sample integer values with the shape of ``input``."""
    from . import (
        _dtype_to_str,
        jt,
    )
    if high is None:
        low, high = 0, low
    result = jt.randint(int(low), int(high), tuple(int(s) for s in input.shape))
    return result.cast(_dtype_to_str(dtype)) if dtype is not None else result
