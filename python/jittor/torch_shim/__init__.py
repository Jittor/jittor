"""Torch-compatibility helpers for running PyTorch-oriented code on Jittor."""

from .bootstrap import (
    NativeExtension,
    build_extension_dirs,
    enable,
    scan_extension_dirs,
)

__all__ = [
    "NativeExtension",
    "build_extension_dirs",
    "enable",
    "scan_extension_dirs",
]
