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


def _wrapped_flash_attn_forward(q, k, v, dropout_p=0.0, softmax_scale=None,
                                causal=False, window_size_left=-1,
                                window_size_right=-1, softcap=0.0,
                                alibi_slopes=None, return_lse=False):
    """The private forward entry point the flash-attn 2 backend calls.

    Two reasons this name has to exist here. `diffusers` imports it (with its
    backward twin) to decide whether the flash-attn backend is usable at all --
    the import sits in a `try/except` that disables flash attention for the
    whole model when it fails, silently and for every later request. And it is
    the call the flash-attn 2 path actually makes, so it keeps upstream's
    argument order and four-element result.
    """
    import torch

    if return_lse:
        # A wrong log-sum-exp is worse than none: it feeds the backward pass and
        # context-parallel reductions, and the shim's kernels do not surface the
        # running statistic. Only training and context parallelism ask for it.
        raise NotImplementedError(
            "the jittor flash-attn shim does not expose the log-sum-exp"
        )
    out = flash_attn_func(
        q, k, v, dropout_p=dropout_p, softmax_scale=softmax_scale, causal=causal,
        window_size=(window_size_left, window_size_right), softcap=softcap,
        alibi_slopes=alibi_slopes,
    )
    # Callers permute the LSE unconditionally (`lse.permute(0, 2, 1)`) even when
    # they did not ask for it, so it has to be a tensor of the right rank; an
    # empty one is never read, and materialising the scores to fill it would
    # cost the quadratic memory flash attention exists to avoid.
    lse = torch.empty(q.shape[0], q.shape[1], 0, dtype=torch.float32,
                      device=q.device)
    return out, lse, None, None


def _wrapped_flash_attn_backward(*args, **kwargs):
    """Upstream's private backward entry point; the shim implements forward only."""
    raise NotImplementedError(
        "the jittor flash-attn shim implements the forward pass only"
    )
