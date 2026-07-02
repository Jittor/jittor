"""Process-local PyTorch compatibility bootstrap for Jittor.

Import this module before importing torch-targeted project code:

    from jittor_torch import enable
    enable(project_root=__file__)

The package intentionally lives outside ``jittor`` so it can configure
environment variables before importing Jittor itself.
"""

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
