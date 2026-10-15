"""Torch tensor arithmetic ownership."""

def foreach_zero_(tensors):
    """Clear detached/no-grad tensor lists through the existing zero_ owner."""
    from importlib import import_module
    owner = import_module(__package__)
    if not isinstance(tensors, (list, tuple)):
        raise TypeError("_foreach_zero_ requires a list or tuple of Tensors")
    if not tensors:
        raise RuntimeError("Tensor list must have at least one tensor.")
    # Validate the whole list before any write. The native zero_ owner does not
    # provide Torch's ZeroBackward edge, so do not silently detach that path.
    for index, tensor in enumerate(tensors):
        if not isinstance(tensor, owner.jt.Var):
            raise TypeError("expected Tensor as element %d" % index)
        if tensor._has_view_provenance():
            raise NotImplementedError("_foreach_zero_ view tensor lists are unsupported")
        if not tensor._storage_is_contiguous():
            raise NotImplementedError("_foreach_zero_ non-contiguous tensor lists are unsupported")
        if tensor.requires_grad and not owner.jt.flags.no_grad:
            if tensor.is_leaf:
                raise RuntimeError("a leaf Variable that requires grad is being used in an in-place operation.")
            raise NotImplementedError("_foreach_zero_ differentiable tensor lists are unsupported")
    with owner.jt.no_grad():
        for tensor in tensors:
            tensor.zero_()
    return tensors


def nan_to_num(input, nan=0.0, posinf=None, neginf=None):
    """Replace NaN with ``nan`` and clamp to the +-inf replacement bounds."""
    from importlib import import_module as _import_module
    _owner = _import_module(__package__)
    upper = _owner._FLOAT32_MAX if posinf is None else posinf
    lower = -_owner._FLOAT32_MAX if neginf is None else neginf
    replaced = _owner.jt.ternary(_owner.jt.isnan(input), _owner.jt.full_like(input, nan), input)
    return replaced.minimum(upper).maximum(lower)


def logaddexp(input, other):
    """Return ``log(exp(input) + exp(other))`` without overflowing exp()."""
    from importlib import import_module as _import_module
    _owner = _import_module(__package__)
    shift = _owner.jt.maximum(input, other)
    return shift + _owner.jt.log(_owner.jt.exp(input - shift) + _owner.jt.exp(other - shift))
