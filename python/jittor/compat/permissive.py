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

Use it only where absence is the expected state and the caller has already
decided nothing below the prefix will run. A namespace that something does
reach into deserves a real implementation or an honest ImportError -- quietly
fabricating one hides the gap instead of exposing it.
"""
import importlib.abc
import importlib.machinery
import os
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


class PermissiveModule(types.ModuleType):
    # An empty __path__ marks this a package, so submodule imports keep
    # descending through the finder rather than stopping here.
    __path__ = []

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return _fabricate(name)


# Every module name a permissive finder actually fabricated in this process.
# Audit hook: run a workload with JITTOR_TORCH_PERMISSIVE_AUDIT=1 and read this
# back to see which import-time references a new library really needs, instead
# of guessing at an allowlist.
_fabricated = set()

# Names refused, with the prefix they fell under. Useful for the same purpose.
_refused = set()


def fabricated_modules():
    """Module names this process answered with a fabricated stub."""
    return set(_fabricated)


def refused_modules():
    """Module names a permissive prefix declined to fabricate."""
    return set(_refused)


def _audit_mode():
    """True when JITTOR_TORCH_PERMISSIVE_AUDIT asks for record-everything mode.

    In audit mode every import under a permissive prefix is fabricated and
    recorded, so a workload can be run once to discover exactly which
    import-time references it needs before a name is added to an allowlist.
    """
    return str(os.environ.get("JITTOR_TORCH_PERMISSIVE_AUDIT", "")).strip().lower() \
        not in ("", "0", "false", "no", "off")


class _PermissiveFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self, prefix, allow=()):
        self.prefix = prefix
        # Exact module names under this prefix that may be fabricated. A name
        # ending in ".*" allows that subtree.
        self.allow = set(allow)

    def add_allowed(self, names):
        self.allow.update(names)

    def _allows(self, fullname):
        if fullname == self.prefix:
            return True
        if fullname in self.allow:
            return True
        for entry in self.allow:
            if entry.endswith(".*") and (
                    fullname == entry[:-2] or fullname.startswith(entry[:-1])):
                return True
        return False

    def find_spec(self, fullname, path, target=None):
        if fullname != self.prefix and not fullname.startswith(self.prefix + "."):
            return None
        if not self._allows(fullname) and not _audit_mode():
            # Not on the known import-time list. Declining here produces a
            # normal ImportError, which is the honest answer: fabricating it
            # would hand back an object that silently returns None from every
            # call. See install_permissive_package's docstring.
            _refused.add(fullname)
            return None
        _fabricated.add(fullname)
        return importlib.machinery.ModuleSpec(fullname, self, is_package=True)

    def create_module(self, spec):
        return PermissiveModule(spec.name)

    def exec_module(self, module):
        pass


__all__ = ["PermissiveModule", "install_permissive_package",
           "fabricated_modules", "refused_modules"]


def install_permissive_package(prefix, meta_path, allow=(), transaction=None):
    """Answer a KNOWN list of unresolved imports under ``prefix`` with a stub.

    ``allow`` is the list of module names that libraries reference at import
    time while gating their *use* behind a runtime check this backend never
    passes. Everything else under the prefix gets a normal ImportError: it used
    to be answered too, so ``from torch.fx.passes.shape_prop import ShapeProp``
    succeeded, ``ShapeProp(...)`` constructed, and calling it returned None --
    a whole analysis pass that silently did nothing.

    An entry ending in ``.*`` allows that subtree.

    Modules already published keep their real implementation: the import
    machinery consults its module table before any finder, so this only ever
    fills in names nothing else provides. Installing twice widens the existing
    finder's allowlist rather than adding a second one.
    """
    for finder in meta_path:
        if isinstance(finder, _PermissiveFinder) and finder.prefix == prefix:
            old_allow = set(finder.allow)
            finder.add_allowed(allow)
            if transaction is not None:
                def restore_allow(f=finder, old=old_allow):
                    f.allow.clear()
                    f.allow.update(old)
                transaction.record_undo(restore_allow)
            return
    finder = _PermissiveFinder(prefix, allow)
    meta_path.insert(0, finder)
    if transaction is not None:
        transaction.record_undo(
            lambda f=finder: meta_path.remove(f) if f in meta_path else None)
