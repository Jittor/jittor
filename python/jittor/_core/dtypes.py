"""Native dtype names at Python, NumPy and code-generation boundaries."""

_python_dtype_types = ()


def register_dtype_type(dtype_type):
    """Register the same explicit frontend type at both argument boundaries."""
    global _python_dtype_types
    import jittor_core
    jittor_core._register_python_dtype_type(dtype_type)
    if dtype_type not in _python_dtype_types:
        _python_dtype_types += (dtype_type,)


def is_dtype(value):
    if isinstance(value, (str,) + _python_dtype_types):
        return True
    import sys
    core = sys.modules.get("jittor_core")
    native_type = getattr(core, "NanoString", None)
    return native_type is not None and isinstance(value, native_type)


def dtype_name(value):
    """Return a canonical name for metadata; this does not enable computation."""
    if value is None:
        return None
    name = getattr(value, "name", None)
    if not isinstance(name, str):
        name = getattr(value, "__name__", None)
    if not isinstance(name, str):
        name = str(value)
    if name.startswith("torch."):
        name = name[6:]
    return {"float": "float32", "double": "float64", "half": "float16",
            "long": "int64", "short": "int16", "int": "int32",
            "cfloat": "complex64", "cdouble": "complex128"}.get(name, name)


def dtype_for_compute(value):
    """Validate through the same converter used by native op arguments."""
    if value is None:
        return None
    import jittor_core
    return jittor_core._checked_dtype_name(value)
