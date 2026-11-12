"""Torch numerical distance operations."""

def _cdist_impl(x1, x2, p=2.0, compute_mode=None, **kwargs):
    from . import (
        jt,
    )
    from ...types import _dtype_to_str
    diff = x1.unsqueeze(-2) - x2.unsqueeze(-3)
    if p == 0 or p == float("inf"):
        class BoundaryDistance(jt.Function):
            def execute(self, delta):
                self.delta = delta
                self.zero_gradient = p == 0 or delta.shape[-1] == 0
                if delta.shape[-1] == 0:
                    # Empty feature vectors have distance zero for either order.
                    return jt.sum(delta, dim=-1)
                if p == 0:
                    result = jt.sum((delta != 0).int32(), dim=-1).cast(
                        _dtype_to_str(x1.dtype))
                    if not delta.is_stop_grad():
                        result.start_grad()
                    return result
                self.distance = jt.amax(jt.abs(delta), dim=-1)
                return self.distance

            def grad(self, grad_output):
                if self.zero_gradient:
                    # Keep p=0 attached to its inputs, with a defined zero VJP.
                    return jt.zeros_like(self.delta)
                # cdist gives every tied maximum its full derivative; amax
                # instead divides the incoming gradient between tied entries.
                selected = (jt.abs(self.delta) == self.distance.unsqueeze(-1))
                dtype = _dtype_to_str(x1.dtype)
                sign = (self.delta > 0).cast(dtype) - (self.delta < 0).cast(dtype)
                return grad_output.unsqueeze(-1) * selected * sign

        return BoundaryDistance.apply(diff)
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
    # Torch permits the boundary vector to originate on the default device;
    # compare on the input placement so CUDA multimodal position grids do not
    # mix backends during the broadcasted comparison.
    if hasattr(input, "is_cuda") and hasattr(boundaries, "is_cuda"):
        if bool(input.is_cuda) != bool(boundaries.is_cuda):
            boundaries = boundaries.to("cuda" if input.is_cuda else "cpu")
    flattened = boundaries.reshape((-1,))
    comparison = ((input.unsqueeze(-1) >= flattened)
                  if right else (input.unsqueeze(-1) > flattened))
    result = comparison.int32().sum(-1)
    # Integer sums follow Torch's int64 accumulation default. Select the
    # requested index width after that reduction, including out_int32=True.
    return result.int32() if out_int32 else result.int64()


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
