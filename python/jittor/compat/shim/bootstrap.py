"""Stable public bootstrap surface backed by family-owned shim modules."""

from .build import build_extension_dirs
from .discovery import NativeExtension, scan_extension_dirs
from .runtime import activate, activation_status
from jittor.compat.torch.namespace import (
    TorchNamespace, independent_torch_namespace,
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
]
