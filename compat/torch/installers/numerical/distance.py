"""Torch numerical distance operations."""

def _cdist_impl(x1, x2, p=2.0, compute_mode=None, **kwargs):
    from . import (
        jt,
    )
    diff = x1.unsqueeze(-2) - x2.unsqueeze(-3)
    if p == 2:
        return jt.sqrt((diff * diff).sum(-1))
    if p == 1:
        return jt.abs(diff).sum(-1)
    return (jt.abs(diff) ** p).sum(-1) ** (1.0 / p)


def cdist(x1, x2, p=2.0, compute_mode=None, **kwargs):
    """Return pairwise distances between the rows of two tensors."""
    from . import (
        _cdist_impl,
    )
    return _cdist_impl(
        x1, x2, p=p, compute_mode=compute_mode, **kwargs)


def _bucketize_impl(
        input, boundaries, out_int32=False, right=False, **kwargs):
    flattened = boundaries.reshape((-1,))
    comparison = ((input.unsqueeze(-1) >= flattened)
                  if right else (input.unsqueeze(-1) > flattened))
    result = comparison.int32().sum(-1)
    return result if out_int32 else result.int64()


def bucketize(input, boundaries, out_int32=False, right=False, **kwargs):
    """Return insertion indices for values in sorted boundaries."""
    from . import (
        _bucketize_impl,
    )
    return _bucketize_impl(
        input, boundaries, out_int32=out_int32, right=right, **kwargs)


def _pdist_impl(input, p=2.0):
    from . import (
        jt,
    )
    size = int(input.shape[0])
    differences = input.unsqueeze(1) - input.unsqueeze(0)
    distances = ((jt.abs(differences) ** p).sum(-1)) ** (1.0 / p)
    rows = [i for i in range(size) for _ in range(i + 1, size)]
    cols = [j for i in range(size) for j in range(i + 1, size)]
    return distances[jt.array(rows), jt.array(cols)]


def pdist(input, p=2.0):
    """Return pairwise p-norm distances between rows of a tensor."""
    from . import (
        _pdist_impl,
    )
    return _pdist_impl(input, p=p)
