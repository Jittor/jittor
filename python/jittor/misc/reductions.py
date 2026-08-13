"""Small native reduction aliases retained for the historical root API."""

import jittor as jt


def amax(input, dim=None, keepdim=False, keepdims=None):
    """Return values-only maximum reduction (the native Jittor contract)."""
    if keepdims is not None:
        keepdim = keepdims
    if dim is None:
        return input.max()
    return jt.max(input, dim, keepdims=keepdim)


def amin(input, dim=None, keepdim=False, keepdims=None):
    """Return values-only minimum reduction (the native Jittor contract)."""
    if keepdims is not None:
        keepdim = keepdims
    if dim is None:
        return input.min()
    return jt.min(input, dim, keepdims=keepdim)


def count_nonzero(input, dim=None):
    """Count non-zero entries without requiring Torch compatibility mode."""
    values = (input != 0).int32()
    return values.sum(dim) if dim is not None else values.sum()


__all__ = ["amax", "amin", "count_nonzero"]
