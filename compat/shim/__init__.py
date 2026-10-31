"""Lazy public API for the canonical Jittor Torch shim domain."""

from __future__ import absolute_import

import importlib


_PUBLIC = {
    "NativeExtension": ("jittor.compat.shim.discovery", "NativeExtension"),
    "scan_extension_dirs": ("jittor.compat.shim.discovery", "scan_extension_dirs"),
    "build_extension_dirs": ("jittor.compat.shim.build", "build_extension_dirs"),
    "activate": ("jittor.compat.shim.runtime", "activate"),
    "activation_status": ("jittor.compat.shim.runtime", "activation_status"),
    "enable": ("jittor.compat.shim.runtime", "activate"),
    "TorchNamespace": ("jittor.compat.torch.publication", "TorchNamespace"),
    "independent_torch_namespace": (
        "jittor.compat.torch.publication", "independent_torch_namespace"
    ),
    "publish_independent_namespace": (
        "jittor.compat.torch.publication", "publish_independent_namespace"
    ),
    "namespace_owner": (
        "jittor.compat.torch.publication", "namespace_owner"
    ),
    "bind_published_namespace": (
        "jittor.compat.torch.publication", "bind_published_namespace"
    ),
    "DISTRIBUTION_ROOT": (
        "jittor.compat.torch.distribution", "DISTRIBUTION_ROOT"
    ),
    "DISTRIBUTION_PROJECT": (
        "jittor.compat.torch.distribution", "DISTRIBUTION_PROJECT"
    ),
    "DISTRIBUTION_SCHEMA_VERSION": (
        "jittor.compat.torch.distribution", "DISTRIBUTION_SCHEMA_VERSION"
    ),
    "DISTRIBUTION_IMPORT_ROOT": (
        "jittor.compat.torch.distribution", "DISTRIBUTION_IMPORT_ROOT"
    ),
    "DISTRIBUTION_MODULES": (
        "jittor.compat.torch.distribution", "DISTRIBUTION_MODULES"
    ),
    "DISTRIBUTION_PACKAGE_ALIASES": (
        "jittor.compat.torch.distribution", "DISTRIBUTION_PACKAGE_ALIASES"
    ),
    "distribution_module_names": (
        "jittor.compat.torch.distribution", "distribution_module_names"
    ),
    "distribution_manifest": (
        "jittor.compat.torch.distribution", "distribution_manifest"
    ),
    "distribution_metadata": (
        "jittor.compat.torch.distribution", "distribution_metadata"
    ),
    "distribution_package_names": (
        "jittor.compat.torch.distribution", "distribution_package_names"
    ),
    "validate_distribution_aliases": (
        "jittor.compat.torch.distribution", "validate_distribution_aliases"
    ),
    "validate_distribution_manifest": (
        "jittor.compat.torch.distribution", "validate_distribution_manifest"
    ),
    "validate_distribution_metadata": (
        "jittor.compat.torch.distribution", "validate_distribution_metadata"
    ),
    "validate_distribution_graph": (
        "jittor.compat.torch.distribution", "validate_distribution_graph"
    ),
    "validate_distribution_publication": (
        "jittor.compat.torch.distribution", "validate_distribution_publication"
    ),
    "validate_distribution_boundary": (
        "jittor.compat.torch.distribution", "validate_distribution_boundary"
    ),
    "validate_distribution_bootstrap": (
        "jittor.compat.torch.distribution", "validate_distribution_bootstrap"
    ),
}

__all__ = sorted(_PUBLIC)


def __getattr__(name):
    target = _PUBLIC.get(name)
    if target is None:
        raise AttributeError("module %r has no attribute %r" % (__name__, name))
    try:
        module = importlib.import_module(target[0])
    except AttributeError as error:
        # `from jittor.compat.shim import X` reads an AttributeError out of
        # here as "X does not exist" and replaces it with a bare "cannot
        # import name 'X'", dropping the error that actually stopped the
        # module from importing.
        raise ImportError("importing %s for %s.%s failed: %s: %s" % (
            target[0], __name__, name, type(error).__name__, error)) from error
    value = getattr(module, target[1])
    globals()[name] = value
    return value
