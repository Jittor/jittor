"""Restricted pickle globals shared by portable and Torch archive readers."""
import pickle as _pickle

_SAFE_NUMPY_NAMES = frozenset((
    "ndarray", "dtype", "_reconstruct", "scalar",
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
