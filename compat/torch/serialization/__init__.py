"""Torch serialization composed from module-owned codecs and readers."""
from contextlib import ContextDecorator

from .portable import load, save
from .safetensors import _install_safetensors_shim as _install_safetensors_shim
from ..fidelity import Fidelity, register_api_bindings


# PyTorch 2.6 exposes a process-local allow-list used by its restricted
# weights-only unpickler. The portable loader has its own explicit safe
# unpickling policy, but Accelerate temporarily manages this list while it
# restores optimizer state. Keep the same observable API without making the
# list influence the portable decoder.
_SAFE_GLOBALS = []


def get_safe_globals():
    return list(_SAFE_GLOBALS)


def clear_safe_globals():
    _SAFE_GLOBALS.clear()


def add_safe_globals(safe_globals):
    for item in safe_globals:
        if item not in _SAFE_GLOBALS:
            _SAFE_GLOBALS.append(item)


class safe_globals(ContextDecorator):
    def __init__(self, safe_globals):
        self._safe_globals = list(safe_globals)
        self._previous = None

    def __enter__(self):
        self._previous = get_safe_globals()
        add_safe_globals(self._safe_globals)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        _SAFE_GLOBALS[:] = self._previous
        return False


def install(ctx):
    g = ctx.target_namespace
    g.save = save
    g.load = load
    g._vj_pickle_load = load
    g._vj_pickle_save = save
    # Accelerate imports this as ``torch.serialization`` rather than through
    # the standalone ``torch.serialization`` module path.
    g.serialization = __import__(__name__, fromlist=["*"])
    register_api_bindings(g, "torch", ("save", "load"), Fidelity.APPROXIMATE,
        "Portable tensor pickle and supported Torch zip storages; restricted "
        "unpickling is the default, native-only formats require explicit unsafe opt-in")


__all__ = [
    "load", "save", "install", "get_safe_globals", "clear_safe_globals",
    "add_safe_globals", "safe_globals",
]
