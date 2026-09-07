"""Torch numerical shape operations."""

def _vstack_impl(tensors):
    from . import (
        jt,
    )
    tensors = list(tensors)
    return jt.concat(
        [t if t.ndim >= 2 else t.reshape((1, -1)) for t in tensors], dim=0)


def vstack(tensors):
    """Stack tensors vertically using the Torch-compatible shape rules."""
    from . import (
        _vstack_impl,
    )
    return _vstack_impl(tensors)


def row_stack(tensors):
    """Alias of :func:`vstack` with Torch's historical spelling."""
    from . import (
        _vstack_impl,
    )
    return _vstack_impl(tensors)


def hstack(tensors):
    """Stack one-dimensional tensors along 0 and higher-rank tensors along 1."""
    from . import (
        _py_all,
        jt,
    )
    tensors = list(tensors)
    dim = 0 if _py_all(t.ndim == 1 for t in tensors) else 1
    return jt.concat(tensors, dim=dim)


def dstack(tensors):
    """Stack tensors along the third dimension."""
    from . import (
        jt,
    )
    out = []
    for t in list(tensors):
        out.append(t.reshape((1, -1, 1)) if t.ndim == 1
                   else (t.unsqueeze(-1) if t.ndim == 2 else t))
    return jt.concat(out, dim=2)


def column_stack(tensors):
    """Stack one-dimensional tensors as columns."""
    from . import (
        jt,
    )
    tensors = list(tensors)
    return jt.concat(
        [t.reshape((-1, 1)) if t.ndim == 1 else t for t in tensors], dim=1)


def _movedim_impl(x, source, destination):
    nd = x.ndim
    src = [s % nd for s in (
        source if isinstance(source, (list, tuple)) else [source])]
    dst = [d % nd for d in (
        destination if isinstance(destination, (list, tuple)) else [destination])]
    order = [d for d in range(nd) if d not in src]
    for d, s in sorted(zip(dst, src)):
        order.insert(d, s)
    return x.permute(order)


def movedim(x, source, destination):
    """Move tensor dimensions using Torch-compatible axis numbering."""
    from . import (
        _movedim_impl,
    )
    return _movedim_impl(x, source, destination)


def moveaxis(x, source, destination):
    """Alias of :func:`movedim` with NumPy-compatible naming."""
    from . import (
        _movedim_impl,
    )
    return _movedim_impl(x, source, destination)


def _unflatten_impl(input, dim, sizes):
    d = dim % input.ndim
    return input.reshape(
        list(input.shape[:d]) + list(sizes) + list(input.shape[d + 1:]))


def unflatten(input, dim, sizes):
    """Unflatten one tensor dimension according to Torch shape rules."""
    from . import (
        _unflatten_impl,
    )
    return _unflatten_impl(input, dim, sizes)


def _swapaxes_impl(input, axis0, axis1):
    perm = list(range(input.ndim))
    a, b = axis0 % input.ndim, axis1 % input.ndim
    perm[a], perm[b] = perm[b], perm[a]
    return input.permute(perm)


def swapaxes(input, axis0, axis1):
    """Swap two tensor dimensions."""
    from . import (
        _swapaxes_impl,
    )
    return _swapaxes_impl(input, axis0, axis1)


def swapdims(input, axis0, axis1):
    """Alias of :func:`swapaxes`."""
    from . import (
        _swapaxes_impl,
    )
    return _swapaxes_impl(input, axis0, axis1)


def _ravel_impl(input):
    return input.reshape((-1,))


def ravel(input):
    """Flatten a tensor to one dimension."""
    from . import (
        _ravel_impl,
    )
    return _ravel_impl(input)


def _narrow_impl(input, dim, start, length):
    return input.narrow(dim, start, length)


def narrow(input, dim, start, length):
    """Return a length-sized slice along ``dim`` starting at ``start``."""
    from . import (
        _narrow_impl,
    )
    return _narrow_impl(input, dim, start, length)


def _tile_impl(input, *dims):
    return input.tile(*dims)


def tile(input, *dims):
    """Repeat tensor dimensions using Torch-compatible tile semantics."""
    from . import (
        _tile_impl,
    )
    return _tile_impl(input, *dims)
