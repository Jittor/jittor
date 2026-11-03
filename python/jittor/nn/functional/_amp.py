"""Compute-dtype preparation for native mixed-precision contractions.

`torch.autocast` casts an operator's *inputs* to the compute dtype -- bias
included -- so `Conv3d(float32 x, float32 w, float32 b)` under
`autocast(float16)` computes and returns float16. Jittor's amp register only
biases the dtype an operator infers for its *result*; the bias is added by a
separate `add` after it, and the torch shim's torch-parity promotion
(`float16 + float32 -> float32`) then lifts the whole operator back to
float32. Native jittor's own promotion reads the register and stays in
float16, so this is a shim-versus-core divergence rather than a torch one --
and a fused operator in torch keeps one dtype here.

Every operator that adds its bias itself (linear, the convolution functionals
and their CUDA adapters) therefore casts a wider bias to the product's dtype
while a register that selects a compute dtype is active.

Measured 2026-09-19 on the MiniMax-H3 video VAE decode: without this, all 63
`Conv3d` calls of the decode returned float32 under `autocast(float16)` and the
float32 propagated into the layer norms and linears behind them -- that is the
1.09 s non-attention half of the decode's gap against torch.
"""

import jittor as jt
from jittor._core.dtypes import dtype_name


def amp_contraction_inputs(x, weight):
    """Round operands before building a portable AMP forward/backward graph.

    An output-only dtype preference leaves original FP32 operands saved for
    the backward. cuDNN casts both operands before invoking its kernel; the
    portable contractions need the same input-rounding boundary.
    """
    preferences = (jt.amp_flags.prefer16 | jt.amp_flags.prefer_bfloat16
                   | jt.amp_flags.prefer32)
    if not int(jt.flags.amp_reg) & preferences:
        return x, weight
    compute = str(jt.binary_dtype_infer('add', dtype_name(x.dtype), dtype_name(weight.dtype)))
    return (x if dtype_name(x.dtype) == compute else x.cast(compute),
            weight if dtype_name(weight.dtype) == compute else weight.cast(compute))


def amp_contraction_scope():
    """Use the rounded inputs' dtype without lowering derivative intermediates.

    Call after ``amp_contraction_inputs``. Keep explicit native accumulator
    options, while clearing the preference already applied to the operands.
    """
    preferences = (jt.amp_flags.prefer16 | jt.amp_flags.prefer_bfloat16
                   | jt.amp_flags.prefer32)
    register = (int(jt.flags.amp_reg) & ~preferences) | jt.amp_flags.keep_reduce
    return jt.flag_scope(amp_reg=register)


def prefers_half():
    """Whether the active amp register retypes float operators to a half dtype.

    Read off ``amp_reg``, the register dtype inference actually consults, not
    off ``auto_mixed_precision_level``: ``torch.autocast`` writes ``amp_reg``
    directly and never touches the level. Only the float16/bfloat16 bits count;
    level 3's ``keep_reduce | keep_white`` does not retype anything to a half.

    Both consumers that had their own copy of this test read it for different
    reasons. oneDNN declines a float32 product under a half policy because its
    library always returns float32. LayerNorm builds its statistics and affine
    in float32 (ATen keeps it on the float32 list) and rounds once at the end;
    letting the register lower it to bfloat16's eight mantissa bits gave ~4e-2
    relative error in the statistics and a training trajectory that ended 79%
    off (KI-COMPAT-019).
    """
    return bool(jt.flags.amp_reg
                & (jt.amp_flags.prefer16 | jt.amp_flags.prefer_bfloat16))


def bias_for_compute_dtype(product, bias):
    """``bias`` cast to ``product``'s dtype when the amp register asks for it.

    Returns ``bias`` unchanged when no register is active, which is every call
    of every model that does not use one. The flag reads stay off that path on
    purpose: they measured 23 us per `nn.Linear` call at (1,128,512).
    """
    amp_reg = jt.flags.amp_reg
    if amp_reg and (amp_reg & (jt.amp_flags.prefer16 | jt.amp_flags.prefer32
                             | jt.amp_flags.prefer_bfloat16)) \
            and hasattr(bias, "cast") and product.dtype != bias.dtype:
        return bias.cast(product.dtype)
    return bias
