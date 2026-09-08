"""Torch numerical integration operations."""

def trapz(y, x=None, dx=1, dim=-1, *, out=None):
    """Integrate along a tensor dimension with the trapezoidal rule."""
    from . import (
        _trapz,
    )
    return _trapz(y, x=x, dx=dx, dim=dim, out=out)


def trapezoid(y, x=None, dx=1, dim=-1, *, out=None):
    """Alias of :func:`trapz` using Torch's newer spelling."""
    from . import (
        _trapz,
    )
    return _trapz(y, x=x, dx=dx, dim=dim, out=out)
