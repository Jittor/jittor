"""Shared version and identity contracts for optional adapters."""
import sys


class UnsupportedAdapterVersion(ImportError):
    pass


def require_version(package, supported):
    module = sys.modules.get(package)
    # `getattr`, not `vars(module).get(...)`. A lazily constructed module keeps
    # its attributes behind a module-level `__getattr__` and never puts them in
    # the namespace dict; reading only the dict reports "no version" for a
    # package that has one. Transformers 5.x is exactly that shape -- the module
    # is a `_LazyModule` whose `extra_objects` carry `__version__` -- so the
    # dict lookup returned None and this raised
    # "imported package reports None" for 5.5.3, a version this adapter lists as
    # supported. A real (non-lazy) module answers the same either way.
    version = getattr(module, "__version__", None) if module is not None else None
    # A source checkout's own version wins; never substitute unrelated installed
    # metadata and accidentally approve a different source tree.
    normalized = str(version).split("+", 1)[0]
    if normalized not in supported:
        raise UnsupportedAdapterVersion(
            "%s adapter supports %s; imported package reports %r" %
            (package, ", ".join(sorted(supported)), version))
    return normalized


def required_patch(function):
    function._jittor_required_patch = True
    return function


def replace(owner, name, value, expected):
    from jittor.compat.module_patcher import patch_method
    patch_method(owner, name, value, expected=expected)


def replace_bound_aliases(package, name, original, replacement):
    for path, module in tuple(sys.modules.items()):
        if path != package and not path.startswith(package + "."):
            continue
        if module is not None and vars(module).get(name) is original:
            replace(module, name, replacement, original)
