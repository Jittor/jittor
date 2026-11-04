"""Publish ``torch.export`` over the eager, direct-call implementation.

``torch.export`` is a package with ``exported_program`` and
``dynamic_shapes`` children because importing those module paths is part of
the public API (``from torch.export.exported_program import ExportedProgram``).
The implementation lives in :mod:`jittor.compat.torch.export`; this installer
only builds the module graph and records fidelity.
"""
from __future__ import absolute_import

from .. import export as _export
from ..fidelity import Fidelity, register_api_bindings, register_fidelity


_PUBLIC = ("export", "ExportedProgram", "Constraint", "Dim", "dims")

_SUBMODULES = (
    ("torch.export.exported_program", ("ExportedProgram",)),
    ("torch.export.dynamic_shapes", ("Constraint", "Dim", "dims")),
)

_DETAIL = (
    "export() validates the example inputs by executing them once and returns "
    "a callable that runs the original module eagerly. No ATen graph, lifted "
    "parameters, shape constraints or verifier are produced; graph_module, "
    "graph and graph_signature raise NotImplementedError unless the layer "
    "provides a working torch.fx.symbolic_trace, in which case the traced "
    "GraphModule is exposed for inspection only.")


def install(ctx):
    module = ctx.registry.ensure("torch.export", package=True)
    module.__doc__ = _export.__doc__
    module.__all__ = tuple(_PUBLIC)
    for name in _PUBLIC:
        setattr(module, name, getattr(_export, name))
    for module_name, names in _SUBMODULES:
        child = ctx.registry.ensure(module_name)
        for name in names:
            setattr(child, name, getattr(_export, name))
        if module_name == "torch.export.exported_program":
            module.exported_program = child
        elif module_name == "torch.export.dynamic_shapes":
            module.dynamic_shapes = child
    register_api_bindings(
        module, "torch.export", ("export", "ExportedProgram"),
        Fidelity.APPROXIMATE, _DETAIL)
    register_fidelity(
        "torch.export.export", _export.export, Fidelity.APPROXIMATE, _DETAIL)
    register_fidelity(
        "torch.export.Dim", _export.Dim, Fidelity.APPROXIMATE,
        "Dim carries name/min/max metadata but no shape environment consumes "
        "it: export captures no graph, so dynamic dimensions cannot be "
        "specialized or checked.")
