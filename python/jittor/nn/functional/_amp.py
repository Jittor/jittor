"""One rule for adding a fused operator's bias while the amp register is active.

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


def bias_for_compute_dtype(product, bias):
    """``bias`` cast to ``product``'s dtype when the amp register asks for it.

    Returns ``bias`` unchanged when no register is active, which is every call
    of every model that does not use one. The flag reads stay off that path on
    purpose: they measured 23 us per `nn.Linear` call at (1,128,512).
    """
    amp_reg = jt.flags.amp_reg
    if amp_reg and (amp_reg & (jt.amp_flags.prefer16 | jt.amp_flags.prefer32)) \
            and hasattr(bias, "cast") and product.dtype != bias.dtype:
        return bias.cast(product.dtype)
    return bias
