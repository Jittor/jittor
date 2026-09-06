"""Compatibility entry point for the native ROCm backend.

The implementation is now source-owned by :mod:`jittor.backends.rocm`; this
module only preserves older bootstrap imports.
"""

from jittor.backends.rocm import configure, install_extern, post_process


def install(context):
    """Return the immutable native ROCm build configuration."""
    return configure(context)


def convert_nvcc_flags(flags):
    """ROCm kernels already use HIP flags; keep the legacy hook pure."""
    return flags
