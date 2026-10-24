"""Native dtype names at Python, NumPy and code-generation boundaries."""
import sys
from typing import Tuple

_python_dtype_types: Tuple[type, ...] = ()


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
    # `import sys` used to sit here, on a path `jt.empty` hits once per allocation.
    core = sys.modules.get("jittor_core")
    native_type = getattr(core, "NanoString", None)
    return native_type is not None and isinstance(value, native_type)


#: Spellings that are not canonical jittor dtype names. Hoisted out of
#: dtype_name because that runs a few hundred times per training step and was
#: rebuilding this literal on every call.
_CANONICAL_ALIASES = {
    "float": "float32", "double": "float64", "half": "float16",
    "long": "int64", "short": "int16", "int": "int32",
    "cfloat": "complex64", "cdouble": "complex128",
}


#: Canonical name by the spelling that was read off the value. Both steps that
#: follow the read -- the ``torch.`` prefix and the alias table -- are a pure
#: function of that one string, and this runs a few hundred times per training
#: step. Bounded, because a caller can reach the ``str(value)`` arm with an
#: arbitrary object whose text is not a dtype name at all.
_CANONICAL_NAMES = {}
_CANONICAL_NAMES_LIMIT = 256


#: `jittor_core.NanoString`, learned from the first one seen here. A native
#: dtype has neither attribute the general arm probes for, so it paid two
#: failed lookups before its `str`; indexing asks this of every index Var.
_NATIVE_DTYPE_TYPE = None


def dtype_name(value):
    """Return a canonical name for metadata; this does not enable computation."""
    global _NATIVE_DTYPE_TYPE
    cls = value.__class__
    if cls is str:
        # The overwhelmingly common non-native spelling, and the one the two
        # attribute probes below can never answer for.
        name = value
    elif cls is _NATIVE_DTYPE_TYPE:
        name = str(value)
    elif value is None:
        return None
    else:
        name = getattr(value, "name", None)
        if not isinstance(name, str):
            name = getattr(value, "__name__", None)
            if not isinstance(name, str):
                name = str(value)
                if cls.__name__ == "NanoString" and cls.__module__ == "jittor_core":
                    _NATIVE_DTYPE_TYPE = cls
    canonical = _CANONICAL_NAMES.get(name)
    if canonical is None:
        canonical = name[6:] if name.startswith("torch.") else name
        canonical = _CANONICAL_ALIASES.get(canonical, canonical)
        if len(_CANONICAL_NAMES) < _CANONICAL_NAMES_LIMIT:
            _CANONICAL_NAMES[name] = canonical
    return canonical


#: The native `Var.dtype` getter, resolved on first use. A frontend may wrap
#: `dtype` in a property of its own (the torch frontend does, to hand out
#: `torch.float32`), so the descriptor is taken from where that frontend keeps
#: the original when it has replaced it.
_VAR_DTYPE = None


def var_dtype_name(var):
    """The canonical dtype name of a Var, read natively.

    For code that only compares or passes on the name. Going through
    ``var.dtype`` under the torch frontend runs its Python property and builds
    a ``torch.dtype`` just to turn it back into this string: an optimizer step
    did that about twelve times per parameter.
    """
    getter = _VAR_DTYPE
    if getter is None:
        import jittor_core
        cls = jittor_core.Var
        getter = getattr(cls, "_frontend_native_dtype", None) or cls.__dict__["dtype"]
        globals()["_VAR_DTYPE"] = getter
    if isinstance(var, getter.__objclass__):
        return str(getter.__get__(var))
    # A real independent TensorBase is a sibling, not a Var subclass. Never
    # invoke Var's C descriptor on that sibling or read its frontend dtype table.
    import jittor_core
    from types import GetSetDescriptorType
    if not jittor_core._is_tensor_carrier(var):
        raise TypeError("native dtype requires an initialized tensor carrier")
    cls = type(var)
    getter = getattr(cls, "_frontend_native_dtype", None)
    if getter is None:
        # Raw owned roots have not installed the frontend property. Resolve
        # their own genuine native descriptor through their physical MRO.
        for base in cls.__mro__:
            candidate = vars(base).get("dtype")
            if isinstance(candidate, GetSetDescriptorType):
                getter = candidate
                break
    if (not isinstance(getter, GetSetDescriptorType)
            or getter.__name__ != "dtype"
            or getter.__objclass__ not in cls.__mro__):
        raise TypeError("native dtype descriptor must belong to the actual carrier")
    return str(getter.__get__(var, cls))


def dtype_for_compute(value):
    """Validate through the same converter used by native op arguments."""
    if value is None:
        return None
    import jittor_core
    return jittor_core._checked_dtype_name(value)
