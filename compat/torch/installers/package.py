"""Publish ``torch.package`` over the data-only archive implementation.

The implementation lives in :mod:`jittor.compat.torch.package`; this installer
only creates the public module graph.  ``torch.package`` is a package (as it is
upstream) with ``package_exporter``, ``package_importer`` and ``importer``
children, because importing those names directly is part of the public API and
of what ``torch.fx.graph_module`` does at import time.
"""
from __future__ import absolute_import

from .. import package as _package
from ..fidelity import Fidelity, register_api_bindings


_PUBLIC = (
    "PackageExporter", "PackageImporter", "Importer", "OrderedImporter",
    "sys_importer", "ObjNotFoundError", "ObjMismatchError",
    "PackagingError", "EmptyMatchError",
)

_SUBMODULES = (
    ("torch.package.package_exporter",
     ("PackageExporter", "PackagingError", "EmptyMatchError")),
    ("torch.package.package_importer",
     ("PackageImporter", "_package_imported_modules")),
    ("torch.package.importer",
     ("Importer", "OrderedImporter", "sys_importer",
      "ObjNotFoundError", "ObjMismatchError")),
)

_DETAIL = (
    "Zip archive of data resources written and read with torch.save; Python "
    "source packaging (intern/mock/extern/save_source_string/save_module/"
    "import_module) raises NotImplementedError, so archives are not hermetic, "
    "and they are not binary-compatible with PyTorch's torch.package format.")


def install(ctx):
    module = ctx.registry.ensure("torch.package", package=True)
    module.__doc__ = _package.__doc__
    module.__all__ = tuple(_PUBLIC)
    for name in _PUBLIC:
        setattr(module, name, getattr(_package, name))
    for module_name, names in _SUBMODULES:
        child = ctx.registry.ensure(module_name)
        for name in names:
            setattr(child, name, getattr(_package, name))
    register_api_bindings(
        module, "torch.package", ("PackageExporter", "PackageImporter"),
        Fidelity.APPROXIMATE, _DETAIL)
