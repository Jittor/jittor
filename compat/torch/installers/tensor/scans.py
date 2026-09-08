"""Torch tensor scans ownership."""
from jittor._core.dtypes import dtype_name as _jittor_dtype_name

def _assign_out(out, value):
    """Assign through the native view owner, including retained output views."""
    from importlib import import_module as _import_module
    _owner = _import_module(__package__)
    out.assign(value)
    return out


def _cumulative(native, input, dim, dtype, out):
    # ACL's aclnnCumsum SEGFAULTs on bool input (transformers builds position_ids
    # via mask.cumsum(-1)); Torch promotes bool/uint8 to int64 anyway, so casting
    # first matches Torch and dodges the crash.
    from importlib import import_module as _import_module
    _owner = _import_module(__package__)
    if isinstance(input, _owner.jt.Var) and _jittor_dtype_name(input.dtype) in ("bool", "uint8"):
        input = input.cast("int64")
    result = native(input, dim)
    if dtype is not None:
        result = result.cast(_owner._dtype_to_str(dtype))
    if out is not None:
        return _owner._assign_out(out, result)
    return result


def cumsum(input, dim=-1, dtype=None, out=None, axis=None, **kwargs):
    """Return a cumulative sum along ``dim`` with Torch's dtype promotion."""
    from importlib import import_module as _import_module
    _owner = _import_module(__package__)
    return _owner._cumulative(
        _owner._NATIVE_CUMSUM, input, dim if axis is None else axis, dtype, out)


def cumprod(input, dim=-1, dtype=None, out=None, axis=None, **kwargs):
    """Return a cumulative product along ``dim`` with Torch's dtype promotion."""
    from importlib import import_module as _import_module
    _owner = _import_module(__package__)
    if _owner._NATIVE_CUMPROD is None:
        raise RuntimeError("torch.cumprod has no native Jittor owner here")
    return _owner._cumulative(
        _owner._NATIVE_CUMPROD, input, dim if axis is None else axis, dtype, out)
