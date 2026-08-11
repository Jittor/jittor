"""Post-core composition of Jittor compatibility domains."""

from __future__ import absolute_import

import sys
from dataclasses import dataclass

from ._aliases import (
    install_aliases,
    publish_loaded_aliases,
    torch_namespace_claimable,
    torch_namespace_owned,
)


@dataclass(frozen=True)
class CompositionReport:
    torch_reports: tuple
    integrations: dict
    aliases: dict


def compose(root_module, core_flags, strict=True, preflight=None):
    """Install required compatibility surfaces after the native core is ready."""

    existing = getattr(root_module, "_compat_composition_report", None)
    if isinstance(existing, CompositionReport):
        if not torch_namespace_owned(root_module):
            raise RuntimeError(
                "completed compatibility module graph was changed after install"
            )
        publish_loaded_aliases(root_module)
        return existing

    aliases = install_aliases(root_module)
    torch_reports = ()
    if torch_namespace_claimable(root_module):
        from . import torch as torch_compat

        torch_compat.install(root_module, strict=strict)
        context = getattr(root_module, "_torch_compat_install_context")
        torch_reports = tuple(context.reports)

    # The canonical Triton domain has always been part of plain Jittor startup:
    # its own idempotent installer owns bare ``import triton`` registration.
    # External backend entry points remain exclusive to explicit shim enable.
    from . import triton as triton_compat  # noqa: F401

    from .shim.control import wrap_flags

    wrap_flags(root_module, core_flags, strict=strict)

    if torch_namespace_claimable(root_module):
        sys.modules["torch"] = root_module
    publish_loaded_aliases(root_module)
    report = CompositionReport(torch_reports, {}, aliases)
    root_module._compat_preflight_result = preflight
    root_module._compat_composition_report = report
    return report
