"""Torch tensor arithmetic ownership."""

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
