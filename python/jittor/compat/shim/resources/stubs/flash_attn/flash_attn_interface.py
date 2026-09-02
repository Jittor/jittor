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


def _unsupported_context_parallel(name):
    def stub(*args, **kwargs):
        raise NotImplementedError(
            f"flash_attn.flash_attn_interface.{name} backs context-parallel attention, which "
            "Jittor does not provide: torch.distributed here has neither point-to-point "
            "communication nor sub-groups. Single-device attention goes through "
            "`flash_attn_func` and does not reach this path."
        )

    stub.__name__ = name
    return stub


# These exist so that `from flash_attn.flash_attn_interface import _wrapped_flash_attn_forward,
# _wrapped_flash_attn_backward` succeeds. Diffusers imports the pair alongside `flash_attn_func`
# and treats an ImportError as "no flash-attn at all", which would silently drop every caller
# back to quadratic-memory attention -- so their absence disables far more than the
# context-parallel path that actually calls them.
_wrapped_flash_attn_forward = _unsupported_context_parallel("_wrapped_flash_attn_forward")
_wrapped_flash_attn_backward = _unsupported_context_parallel("_wrapped_flash_attn_backward")
