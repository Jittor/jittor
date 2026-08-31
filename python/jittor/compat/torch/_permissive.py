"""Import-time stand-ins for torch internals that have no counterpart here.

Upstream torch carries subsystems this shim has no equivalent of -- the
inductor compiler above all. Libraries built on torch import from those
namespaces at module scope while guarding the *use* behind a runtime check, so
on the shim the import fails long before the guard that would have skipped it
ever runs.

A permissive package answers any import below its prefix with a module whose
attributes fabricate themselves, so ``from torch._inductor.codecache import
FxGraphCache`` yields something that can be subclassed, called, or referenced
in an annotation -- and never executed, because the guard behind it is false.

This is confined to torch-internal namespaces (``torch._*``) on purpose. A
public torch API is either implemented here or genuinely missing, and quietly
fabricating one would hide a real gap instead of exposing it.
"""
import importlib.abc
import importlib.machinery
import types


class _FabricatedMeta(type):
    """Metaclass giving a fabricated class no-op callables for any attribute."""

    def __getattr__(cls, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return lambda *args, **kwargs: None


def _fabricate(name):
    return _FabricatedMeta(name, (), {
        "__init__": lambda self, *args, **kwargs: None,
        "__call__": lambda self, *args, **kwargs: None,
    })


class _PermissiveModule(types.ModuleType):
    # An empty __path__ marks this a package, so submodule imports keep
    # descending through the finder rather than stopping here.
    __path__ = []

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return _fabricate(name)


class _PermissiveFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self, prefix):
        self.prefix = prefix

    def find_spec(self, fullname, path, target=None):
        if fullname == self.prefix or fullname.startswith(self.prefix + "."):
            return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
        return None

    def create_module(self, spec):
        return _PermissiveModule(spec.name)

    def exec_module(self, module):
        pass


def install_permissive_package(prefix, meta_path):
    """Answer every unresolved import under ``prefix`` with a fabricated stub.

    Modules already published keep their real implementation: the import
    machinery consults its module table before any finder, so this only ever
    fills in names nothing else provides. Installing twice is a no-op.
    """
    for finder in meta_path:
        if isinstance(finder, _PermissiveFinder) and finder.prefix == prefix:
            return
    meta_path.insert(0, _PermissiveFinder(prefix))
