"""Compatibility re-exports for ``flash_attn.flash_attn_interface``.

The real flash-attn package exposes the public forward helpers both at the
package top level and from this submodule. Keep the Jittor shim behavior aligned
without importing upstream PyTorch or libtorch-backed wheels.
"""

from . import (  # noqa: F401
    flash_attn_func,
    flash_attn_kvpacked_func,
    flash_attn_qkvpacked_func,
    flash_attn_varlen_func,
    flash_attn_varlen_kvpacked_func,
    flash_attn_varlen_qkvpacked_func,
    flashattn_jittor_backend,
    flashattn_jittor_last_error,
    is_flashattn_jittor_available,
)


def _get_block_size_n(*args, **kwargs):
    """Return a conservative block size used by flash-attn tests/utilities."""
    return 128
