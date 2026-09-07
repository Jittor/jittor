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

    def probe_library(self, name, *, load=False):
        """Report why ``get_library`` would answer as it does, without raising.

        ``get_library`` returns ``None`` for "switched off", for "not loaded
        yet" and for "no loader", and *raises* when the loader fails. A caller
        that only looks at the return value therefore cannot tell a library
        that is absent from one whose build is broken -- which is how the cuTT
        build stayed broken while its six tests reported "skipped".

        Returns ``(state, reason, evidence)`` where ``state`` is one of the
        ``CapabilityState`` values as a plain string, so this module stays
        free of the capability layer that consumes it.
        """
        with self._lock:
            enabled = self._enabled.get(name)
            module = self._modules.get(name)
            loader = self._loaders.get(name)
        evidence = {
            "has_loader": loader is not None,
            "has_enabled_policy": enabled is not None,
            "module_loaded": module is not None,
        }

        if enabled is not None:
            try:
                permitted = bool(enabled())
            except Exception as exc:
                return "failed", (
                    "the enabled-policy for %s raised %s: %s"
                    % (name, type(exc).__name__, exc)), evidence
            evidence["enabled_policy"] = permitted
            if not permitted:
                return "disabled", (
                    "%s has an enabled-policy and it currently says no, so "
                    "get_library(%r) reports None even if the library is "
                    "installed" % (name, name)), evidence

        if module is not None:
            ops = getattr(module, "ops", None)
            evidence["has_ops"] = ops is not None
            if ops is None:
                return "failed", (
                    "%s is loaded but exposes no ops module; something built "
                    "it half-way" % name), evidence
            return "available", "%s is loaded and exposes its ops" % name, evidence

        if loader is None:
            return "absent", (
                "no loader is registered for %s, so nothing in this build can "
                "ever produce it" % name), evidence

        if not load:
            return "unprobed", (
                "%s is built on first use and nothing has asked for it yet; "
                "ask again with load=True to find out (that may compile)"
                % name), evidence

        try:
            self.get_library(name, load=True)
        except BaseException as exc:
            evidence["load_error"] = "%s: %s" % (type(exc).__name__, exc)
            return "failed", (
                "%s was requested and its loader raised %s: %s -- this is a "
                "broken build, not a missing library"
                % (name, type(exc).__name__, exc)), evidence

        with self._lock:
            module = self._modules.get(name)
        evidence["module_loaded"] = module is not None
        if module is None:
            return "failed", (
                "%s was requested with load=True, its loader returned without "
                "raising, and no module was published: the loader bailed out "
                "silently" % name), evidence
        ops = getattr(module, "ops", None)
        evidence["has_ops"] = ops is not None
        if ops is None:
            return "failed", (
                "%s loaded but exposes no ops module" % name), evidence
        return "available", "%s loaded on request and exposes its ops" % name, evidence


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


def probe_library(name, *, load=False):
    return _libraries.probe_library(name, load=load)


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
