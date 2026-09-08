"""Stable public bootstrap surface backed by family-owned shim modules."""

from .build import build_extension_dirs
from .discovery import NativeExtension, scan_extension_dirs
from .runtime import activate, activation_status
from jittor.compat.torch.publication import (
    TorchNamespace, independent_torch_namespace, namespace_owner,
    bind_published_namespace, publish_independent_namespace,
)
from jittor.compat.torch.distribution import (
    DISTRIBUTION_PROJECT, DISTRIBUTION_SCHEMA_VERSION, DISTRIBUTION_IMPORT_ROOT,
    DISTRIBUTION_ROOT, DISTRIBUTION_MODULES, DISTRIBUTION_PACKAGE_ALIASES,
    distribution_manifest, distribution_metadata, distribution_module_names,
    distribution_package_names,
    validate_distribution_aliases, validate_distribution_graph,
    validate_distribution_manifest, validate_distribution_metadata,
    validate_distribution_publication, validate_distribution_boundary,
    validate_distribution_bootstrap,
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
    "bind_published_namespace",
    "publish_independent_namespace",
    "DISTRIBUTION_ROOT",
    "DISTRIBUTION_PROJECT",
    "DISTRIBUTION_SCHEMA_VERSION",
    "DISTRIBUTION_IMPORT_ROOT",
    "DISTRIBUTION_MODULES",
    "DISTRIBUTION_PACKAGE_ALIASES",
    "distribution_manifest",
    "distribution_metadata",
    "distribution_module_names",
    "distribution_package_names",
    "validate_distribution_aliases",
    "validate_distribution_manifest",
    "validate_distribution_metadata",
    "validate_distribution_graph",
    "validate_distribution_publication",
    "validate_distribution_boundary",
    "validate_distribution_bootstrap",
]
