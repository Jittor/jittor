"""Post-core composition of Jittor compatibility domains."""

from __future__ import absolute_import

from dataclasses import dataclass

from ._aliases import (
    install_aliases,
    publish_loaded_aliases,
    torch_compat_requested,
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
        if existing.torch_reports and not torch_namespace_owned(root_module):
            raise RuntimeError(
                "completed compatibility module graph was changed after install"
            )
        publish_loaded_aliases(root_module)
        return existing

    aliases = install_aliases(root_module)
    torch_reports = ()
    integrations = {}
    torch_mode = torch_compat_requested(root_module, preflight)
    if torch_mode:
        if not torch_namespace_claimable(root_module):
            raise RuntimeError(
                "cannot install Jittor Torch compatibility over an existing "
                "Torch module graph"
            )
        from .shim.runtime import activate

        activation = activate(
            strict=strict,
            verbose=False,
            _root_module=root_module,
            _preflight_result=preflight,
            _composition=True,
        )
        target = activation.get("torch", root_module) if isinstance(activation, dict) else root_module
        context = vars(target)["_torch_compat_install_context"]
        torch_reports = tuple(context.reports)
        if isinstance(activation, dict):
            integrations = activation.get("integrations") or {}

    # Backend integrations belong to explicit compatibility activation.
    # Plain native startup must work without this optional distribution.
    if torch_mode:
        from . import triton as triton_compat  # noqa: F401

    # Compatibility activation must not wrap the process-global native flags
    # object merely to smuggle in an activation side effect.
    root_module.flags = core_flags
    publish_loaded_aliases(root_module)
    report = CompositionReport(torch_reports, integrations, aliases)
    root_module._compat_preflight_result = preflight
    root_module._compat_composition_report = report
    return report
