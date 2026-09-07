"""Runtime-provided compatibility services, without importing the runtime."""

from types import ModuleType
from typing import Callable, Dict


_module_loaders: Dict[str, Callable[[], ModuleType]] = {}


def register_runtime_module(name, loader):
    """Publish a higher-level module loader during runtime bootstrap."""
    if not callable(loader):
        raise TypeError("runtime module loader must be callable")
    previous = _module_loaders.get(name)
    if previous is not None and previous is not loader:
        raise RuntimeError("runtime module service already registered: " + name)
    _module_loaders[name] = loader


def runtime_module(name):
    loader = _module_loaders.get(name)
    if loader is None:
        raise RuntimeError(
            "jittor_utils.%s requires an initialized Jittor runtime; "
            "import jittor before using this compatibility entry point" % name
        )
    module = loader()
    if not isinstance(module, ModuleType):
        raise TypeError("runtime module loader did not return a module: " + name)
    return module


def module_compatibility(name):
    """Return attribute hooks preserving the canonical function objects."""
    def resolve(attribute):
        if attribute.startswith("__") and attribute != "__all__":
            raise AttributeError(attribute)
        module = runtime_module(name)
        if attribute == "__all__":
            return getattr(module, "__all__", [
                key for key in vars(module) if not key.startswith("_")
            ])
        return getattr(module, attribute)

    def names():
        return dir(runtime_module(name))

    return resolve, names
