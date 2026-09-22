"""Ascend SoC capability queries for the ACL kernels.

The aclnn runner registry is hardware-wide; the installed CANN op package is
not -- an Ascend310P-class chip ships a smaller kernel set than a 910B, and
its cube unit cannot run FP32 with ``KEEP_DTYPE`` at all. Kernel code asks
``kernel_available`` before selecting an aclnn path so a missing kernel turns
into a composite implementation instead of an opaque async failure at launch.
"""

import jittor as jt
from jittor import compiler

_soc_name_cache = None

# Runner names (as registered in acl_op_registry) whose CANN kernel is absent
# from the Ascend310P op package. Ascend310P1/3/7 share the same package, so
# the table matches on the "Ascend310P" prefix. Entries are verified against
# the installed opp and on-device failures, not guessed.
_KERNEL_GAPS = {
    "Ascend310P": frozenset(
        (
            "StridedSliceAssignV2",  # no strided_slice_assign* kernel at all
            "EmbeddingBackward",  # aclnnEmbeddingDenseBackward
            "RotaryPositionEmbeddingGrad",
            "FlashAttention",  # aclnnFlashAttentionScoreV2 (prefill)
            "FlashAttentionBackward",
            "NanToNum",
            # Present in the package but produces corrupt writes on Ascend310P
            # (extra rows leak un-scaled input values); treat as unavailable.
            "IndexPutImplAccumulate",
            # Fixed-window pooling aclnn entries are not implemented for
            # Ascend310P (161002); the adaptive variants are.
            "Maxpool",  # aclnnMaxPool2dWithIndices
            "MaxpoolBackward",  # aclnnMaxPool2dWithIndicesBackward
            "Avgpool",  # aclnnAvgPool2d
            "AvgpoolBackward",  # aclnnAvgPool2dBackward
        )
    ),
}


def soc_name():
    """CANN SoC name (e.g. ``"Ascend310P3"``), or "" when ACL is not active."""
    global _soc_name_cache
    if _soc_name_cache is None:
        name = ""
        if getattr(compiler, "has_acl", False):
            try:
                name = jt.core.backend_device_name("acl") or ""
            except Exception:
                name = ""
        _soc_name_cache = name
    return _soc_name_cache


def kernel_available(runner_name):
    """Whether the installed CANN opp provides ``runner_name`` on this SoC."""
    soc = soc_name()
    for prefix, missing in _KERNEL_GAPS.items():
        if soc.startswith(prefix):
            return runner_name not in missing
    return True


def supports_bfloat16():
    """Whether aclnn accepts DT_BFLOAT16 on this SoC (Ascend310P: no)."""
    return not soc_name().startswith("Ascend310P")


def supports_fp32_cube():
    """Whether the cube unit runs FP32 with KEEP_DTYPE (Ascend310*: no)."""
    return not soc_name().startswith("Ascend310")


def ieee_comparisons():
    """Whether elementwise comparisons obey IEEE for non-finite operands.

    On Ascend310P they do not: ``isnan`` never fires, ``x == +inf`` matches
    {nan, +inf} and ``x == -inf`` matches {nan, -inf}. NaN is the only value
    matching both infinities, which gives a usable classification invariant.
    """
    return not soc_name().startswith("Ascend310P")


def allow_reduced_precision():
    """The FP32 cube opt-in, readable as a flag or the legacy module attr."""
    return bool(
        getattr(jt, "acl_allow_hf32", False)
        or getattr(jt.flags, "acl_allow_hf32", 0)
    )


__all__ = [
    "soc_name",
    "kernel_available",
    "supports_bfloat16",
    "supports_fp32_cube",
    "ieee_comparisons",
    "allow_reduced_precision",
]
