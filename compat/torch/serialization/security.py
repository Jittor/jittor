"""Restricted pickle globals shared by portable and Torch archive readers."""
import pickle as _pickle

import jittor as jt

from ..context import get_install_context


def _user_globals():
    return get_install_context(jt).state.setdefault("serialization_safe_globals", [])


def add_safe_globals(safe_globals):
    entries = list(safe_globals)
    for entry in entries:
        value, path = entry if isinstance(entry, tuple) else (entry, None)
        if path is not None and not isinstance(path, str):
            raise TypeError("safe global alias must be a fully qualified string")
        if not hasattr(value, "__module__") or not hasattr(value, "__qualname__"):
            raise TypeError("safe globals must be functions or classes")
    registered = _user_globals()
    for entry in entries:
        if entry not in registered:
            registered.append(entry)


def get_safe_globals():
    return list(_user_globals())


def clear_safe_globals():
    _user_globals().clear()


class safe_globals:
    """Temporarily allow explicitly trusted classes/functions when loading."""

    def __init__(self, globals):
        self.globals = list(globals)

    def __enter__(self):
        previous = get_safe_globals()
        add_safe_globals(self.globals)
        self.added = [entry for entry in get_safe_globals() if entry not in previous]

    def __exit__(self, *exc):
        registered = _user_globals()
        for entry in self.added:
            if entry in registered:
                registered.remove(entry)
        return False


def _registered_global(module, name):
    path = module + "." + name
    for entry in _user_globals():
        value, alias = entry if isinstance(entry, tuple) else (entry, None)
        if path == (alias or value.__module__ + "." + value.__qualname__):
            return value
    return None


_SAFE_NUMPY_NAMES = frozenset((
    # ``_frombuffer`` is how numpy pickles an array that views a buffer, which
    # is what this layer's own ``torch.save`` writes -- without it the shim
    # could not load its own checkpoints at the default ``weights_only=True``.
    # Like ``_reconstruct`` it only builds an array out of bytes, a dtype and a
    # shape; neither can execute anything from the file.
    "ndarray", "dtype", "_reconstruct", "_frombuffer", "scalar",
    "bool_", "int8", "int16", "int32", "int64", "intp", "longlong",
    "uint8", "uint16", "uint32", "uint64", "uintp", "ulonglong",
    "float16", "float32", "float64", "longdouble",
    "complex64", "complex128", "clongdouble", "bytes_", "str_",
))


_SAFE_GLOBALS = frozenset((
    ("collections", "OrderedDict"), ("collections", "defaultdict"),
    ("collections", "Counter"), ("collections", "deque"),
    ("builtins", "set"), ("builtins", "frozenset"), ("builtins", "list"),
    ("builtins", "dict"), ("builtins", "tuple"), ("builtins", "int"),
    ("builtins", "float"), ("builtins", "complex"), ("builtins", "bool"),
    ("builtins", "str"), ("builtins", "bytes"), ("builtins", "bytearray"),
    ("_codecs", "encode"),
    ("jittor.compat.torch.nested", "_rebuild_var_from_numpy"),
    ("jittor.compat.torch.nested", "_rebuild_nested_tensor"),
    # ``torch.Size`` is implemented by the immutable tuple subclass
    # ``_TorchSize``. Portable checkpoints pickle the concrete class name,
    # so permit that exact value type as well as its public alias below.
    ("jittor.compat.torch.nested", "_TorchSize"),
    # torch.dtype/torch.device are plain value objects here (an immutable dtype
    # and a name/index pair); older checkpoints reference them by name.
    ("jittor.compat.torch.types", "dtype"),
    ("jittor.compat.torch.types", "_restore_dtype"),
    ("jittor.compat.torch.types", "device"),
    ("torch", "dtype"), ("torch", "device"), ("torch", "Size"),
))


def _is_safe_global(module, name):
    if (module, name) in _SAFE_GLOBALS:
        return True
    if module in ("numpy", "numpy.core.multiarray", "numpy._core.multiarray",
                  "numpy.core.numeric", "numpy._core.numeric"):
        return name in _SAFE_NUMPY_NAMES
    return False


def _resolve_global(module, name, weights_only):
    """Import module.name, or refuse -- never fabricate an empty class."""
    registered = _registered_global(module, name) if weights_only else None
    if registered is not None:
        return registered
    if weights_only and not _is_safe_global(module, name):
        raise _pickle.UnpicklingError(
            "Weights only load failed: %s.%s is not an allowed global. "
            "torch.load defaults to weights_only=True; re-run with "
            "torch.load(..., weights_only=False) only if you trust the "
            "file, because that lets the checkpoint execute arbitrary code."
            % (module, name))
    try:
        m = __import__(module, fromlist=[name])
        return getattr(m, name)
    except Exception as exc:
        raise _pickle.UnpicklingError(
            "checkpoint refers to %s.%s, which this interpreter cannot "
            "import (%s). Jittor's torch compatibility layer used to "
            "substitute an empty placeholder class here, which loaded the "
            "checkpoint successfully into objects that held none of the "
            "saved state. Install the package that defines it, or load the "
            "checkpoint with weights_only=True to keep only tensors."
            % (module, name, exc))


class _PortableUnpickler(_pickle.Unpickler):
    """Plain-pickle loader for our own torch.save output."""
    weights_only = True

    def find_class(self, module, name):
        return _resolve_global(module, name, self.weights_only)


def _portable_pickle_load(fh, weights_only):
    up = _PortableUnpickler(fh)
    up.weights_only = bool(weights_only)
    return up.load()
