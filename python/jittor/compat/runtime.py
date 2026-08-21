"""Post-core composition of Jittor compatibility domains."""

from __future__ import absolute_import

import sys
from dataclasses import dataclass

from ._aliases import (
    install_aliases,
    publish_loaded_aliases,
    torch_compat_requested,
    torch_namespace_claimable,
    torch_namespace_owned,
)


_NATIVE_COMPOSITION_ATTR = "_compat_native_composition_in_progress"


@dataclass(frozen=True)
class CompositionReport:
    torch_reports: tuple
    integrations: dict
    aliases: dict


def compose(root_module, core_flags, strict=True, preflight=None):
    """Install required compatibility surfaces after the native core is ready."""

    existing = getattr(root_module, "_compat_composition_report", None)
    if isinstance(existing, CompositionReport):
        if existing.torch_reports and not torch_namespace_owned(root_module):
            raise RuntimeError(
                "completed compatibility module graph was changed after install"
            )
        publish_loaded_aliases(root_module)
        return existing

    aliases = install_aliases(root_module)
    torch_reports = ()
    torch_mode = torch_compat_requested(root_module, preflight)
    if torch_mode:
        if not torch_namespace_claimable(root_module):
            raise RuntimeError(
                "cannot install Jittor Torch compatibility over an existing "
                "Torch module graph"
            )
        from . import torch as torch_compat

        torch_compat.install(root_module, strict=strict)
        context = getattr(root_module, "_torch_compat_install_context")
        torch_reports = tuple(context.reports)

    # Real Triton may import ``torch`` while we probe it. If that name is a
    # deployed Jittor placeholder, it must not re-enter the process-wide Torch
    # installer after this composition has already selected native mode.
    if not torch_mode:
        setattr(root_module, _NATIVE_COMPOSITION_ATTR, True)
    try:
        # The canonical Triton domain has always been part of plain Jittor
        # startup: its idempotent installer owns bare ``import triton``
        # registration. External backend entry points remain exclusive to
        # explicit shim enable.
        from . import triton as triton_compat  # noqa: F401
    finally:
        if not torch_mode:
            delattr(root_module, _NATIVE_COMPOSITION_ATTR)

    from .shim.control import wrap_flags

    wrap_flags(root_module, core_flags, strict=strict)

    if torch_mode:
        sys.modules["torch"] = root_module
    publish_loaded_aliases(root_module)
    report = CompositionReport(torch_reports, {}, aliases)
    root_module._compat_preflight_result = preflight
    root_module._compat_composition_report = report
    return report
