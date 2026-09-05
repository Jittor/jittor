"""Loaded backend libraries and injected loaders, independent of bootstrap."""

from threading import RLock
from types import ModuleType


class BackendLibraries:
    """One publication point for modules, their ops, and build resources."""

    def __init__(self):
        self._modules = {}
        self._loaders = {}
        self._enabled = {}
        self._resources = {}
        self._loading = set()
        self._lock = RLock()

    def register_loader(self, name, loader, *, enabled=None):
        if not callable(loader):
            raise TypeError("backend library loader must be callable")
        if enabled is not None and not callable(enabled):
            raise TypeError("backend library enabled policy must be callable")
        with self._lock:
            self._loaders[name] = loader
            if enabled is not None:
                self._enabled[name] = enabled

    def register(self, name, module):
        with self._lock:
            if module is None:
                self._modules.pop(name, None)
            else:
                self._modules[name] = module

    def register_resources(self, name, **resources):
        with self._lock:
            self._resources.setdefault(name, {}).update(resources)

    def get_library(self, name, *, load=False):
        with self._lock:
            enabled = self._enabled.get(name)
            if enabled is not None and not enabled():
                return None
            module = self._modules.get(name)
            loader = self._loaders.get(name)
            if module is not None or not load or loader is None:
                return module
            if name in self._loading:
                raise RuntimeError("recursive backend library load: " + name)
            self._loading.add(name)
            try:
                loader()
                return self._modules.get(name)
            except BaseException:
                self._modules.pop(name, None)
                raise
            finally:
                self._loading.remove(name)

    def get_library_ops(self, name, *, load=False):
        module = self.get_library(name, load=load)
        return getattr(module, "ops", None) if module is not None else None

    def library_resource(self, name, key):
        with self._lock:
            return self._resources.get(name, {}).get(key)


_libraries = BackendLibraries()


def register_library(name, module):
    _libraries.register(name, module)


def register_library_loader(name, loader, *, enabled=None):
    _libraries.register_loader(name, loader, enabled=enabled)


def register_library_resources(name, **resources):
    _libraries.register_resources(name, **resources)


def get_library(name, *, load=False):
    return _libraries.get_library(name, load=load)


def get_library_ops(name, *, load=False):
    return _libraries.get_library_ops(name, load=load)


def library_resource(name, key):
    return _libraries.library_resource(name, key)


LIBRARY_NAMES = (
    "mkl", "mpi", "nccl", "hccl", "cub", "cutt", "cudnn", "cublas",
    "curand", "cufft", "cusparse",
)
LEGACY_LIBRARY_NAMES = frozenset(
    LIBRARY_NAMES + tuple(name + "_ops" for name in LIBRARY_NAMES)
    + ("hccl_mod", "cub_home")
)
ROOT_LIBRARY_NAMES = frozenset((
    "mkl_ops", "mpi", "mpi_ops", "cudnn", "cublas", "curand", "cufft", "cusparse",
))


def library_attribute(name):
    if name not in LEGACY_LIBRARY_NAMES:
        raise AttributeError(name)
    if name == "cub_home":
        return library_resource("cub", "home")
    if name == "hccl_mod":
        return get_library("hccl")
    if name.endswith("_ops"):
        return get_library_ops(name[:-4])
    return get_library(name)


class _LibraryModule(ModuleType):
    def __setattr__(self, name, value):
        if name in self.__dict__.get("_library_readonly_names", ()):
            raise AttributeError(name + " is a read-only backend library query")
        super().__setattr__(name, value)

    def __delattr__(self, name):
        if name in self.__dict__.get("_library_readonly_names", ()):
            raise AttributeError(name + " is a read-only backend library query")
        super().__delattr__(name)

    def __dir__(self):
        return sorted(set(super().__dir__()) | self._library_readonly_names)


def protect_library_attributes(module, names=LEGACY_LIBRARY_NAMES):
    module._library_readonly_names = frozenset(names)
    module.__class__ = _LibraryModule
