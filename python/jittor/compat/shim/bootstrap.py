"""Stable public bootstrap surface backed by family-owned shim modules."""

from .build import build_extension_dirs
from .discovery import NativeExtension, scan_extension_dirs
from .runtime import activate, activation_status
from jittor.compat.torch.publication import (
    TorchNamespace, independent_torch_namespace, namespace_owner,
)
from jittor.compat.torch.distribution import (
    DISTRIBUTION_MODULES, DISTRIBUTION_PACKAGE_ALIASES,
    distribution_manifest, distribution_module_names, distribution_package_names,
    validate_distribution_aliases, validate_distribution_graph,
    validate_distribution_publication,
)

enable = activate

__all__ = [
    "NativeExtension",
    "activate",
    "activation_status",
    "build_extension_dirs",
    "enable",
    "scan_extension_dirs",
    "TorchNamespace",
    "independent_torch_namespace",
    "namespace_owner",
    "DISTRIBUTION_MODULES",
    "DISTRIBUTION_PACKAGE_ALIASES",
    "distribution_manifest",
    "distribution_module_names",
    "distribution_package_names",
    "validate_distribution_aliases",
    "validate_distribution_graph",
    "validate_distribution_publication",
]
