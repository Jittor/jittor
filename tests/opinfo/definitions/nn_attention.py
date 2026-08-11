# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Attention OpInfos: scaled_dot_product_attention (SDPA) -- the transformer core.

SDPA is the single highest-value *backward* target in the audit: every transformer
training step differentiates ``softmax(Q@K^T*scale [+ mask]) @ V`` w.r.t. Q, K and V,
and the softmax-Jacobian-times-V path was the top untested gradient. This module pins
the forward to an INDEPENDENT numpy reference (adapted verbatim from the validated
``test_torch_compat_attention._sdpa_ref`` / ``_softmax``) and -- via the generic
``test_ops`` driver -- gradchecks the backward in float64 against that reference.

Differentiation contract (see ``test_ops._diff_plan``): the primary ``input`` and any
*floating* positional ``args`` are the differentiated leaves. We pass Q as ``input``
and K, V as positional float Vars, so all three are differentiated (the loss is
genuinely differentiable w.r.t. each). ``is_causal`` / ``scale`` are non-tensor and
passed as kwargs, so they are held fixed.

Op resolution surprise: ``jittor.nn.functional.scaled_dot_product_attention`` is the
fallback installed by ``torch_compat`` (jittor has no native functional SDPA; the
``jittor.attention`` module-level function is *not* re-exported onto ``nn.functional``).
That fallback masks with a large finite ``-1e30`` rather than ``-inf``; in float64 the
masked softmax entries underflow to exactly 0.0, so the ``-inf`` numpy reference here
matches it to tolerance. The mask is a constant additive bias (independent of Q/K/V),
so it is held fixed under finite differencing -- backward is smooth and twice
differentiable (matmul -> softmax -> matmul), hence both gradcheck and gradgradcheck
are exercised.

Sizes are kept tiny (each differentiated tensor <= ~24-32 float64 elements) because
gradcheck is O(numel) forward passes: an unbatched (4, 6) L x E sample and a batched
(1, 1, 4, 6) B x H x L x E sample, plus a causal variant of each.
"""
from ._refs import *  # noqa: F401,F403  (make_tensor, SampleInput, refs, np, jt, nn, F)
from ..core import OpInfo, UnaryUfuncInfo, BinaryUfuncInfo, ReductionOpInfo


# ------------------------------------------------------------------- numpy refs
# Adapted from test_torch_compat_attention._softmax / _sdpa_ref (preserve-assets).

def _softmax_np(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def scaled_dot_product_attention_ref(query, key, value, attn_mask=None,
                                     dropout_p=0.0, is_causal=False, scale=None):
    """Independent numpy oracle for SDPA.

    scale defaults to 1/sqrt(d_k); attn = softmax(Q@K^T*scale [+ mask], axis=-1);
    out = attn @ V. dropout_p is accepted for signature parity but only the
    deterministic dropout_p == 0 path (a no-op) is referenced here.
    """
    q = np.asarray(query)
    k = np.asarray(key)
    v = np.asarray(value)
    d = q.shape[-1]
    sf = (1.0 / np.sqrt(d)) if scale is None else scale
    scores = (q @ np.swapaxes(k, -1, -2)) * sf
    if is_causal:
        Lq, Lk = q.shape[-2], k.shape[-2]
        # lower-triangular keep; -inf strictly above the diagonal (torch convention).
        causal = np.triu(np.full((Lq, Lk), -np.inf, dtype="float64"), 1)
        scores = scores + causal
    if attn_mask is not None:
        m = np.asarray(attn_mask)
        if m.dtype == np.bool_:
            scores = np.where(m, scores, -np.inf)
        else:
            scores = scores + m
    attn = _softmax_np(scores, axis=-1)
    return attn @ v


# --------------------------------------------------------------- sample builders
# Q is `input`; K and V are positional float Vars -> all three differentiated.
# is_causal / scale are kwargs -> held fixed. Deterministic per-tensor seeds.

def sample_sdpa(op_info, device, dtype, requires_grad):
    """Non-causal SDPA, unbatched (L, E) and batched (B, H, L, E)."""
    out = []
    # unbatched: (L=4, E=6) -> 24 differentiated elements per tensor.
    shapes = [(4, 6), (1, 1, 4, 6)]
    for i, s in enumerate(shapes):
        q = make_tensor(*s, dtype=dtype, requires_grad=requires_grad, seed=820 + i)
        k = make_tensor(*s, dtype=dtype, requires_grad=requires_grad, seed=830 + i)
        v = make_tensor(*s, dtype=dtype, requires_grad=requires_grad, seed=840 + i)
        out.append(SampleInput(q, k, v))
    # explicit (non-default) scale on the small unbatched shape.
    out.append(SampleInput(
        make_tensor(4, 6, dtype=dtype, requires_grad=requires_grad, seed=850),
        make_tensor(4, 6, dtype=dtype, requires_grad=requires_grad, seed=851),
        make_tensor(4, 6, dtype=dtype, requires_grad=requires_grad, seed=852),
        scale=0.25))
    return out


def sample_sdpa_causal(op_info, device, dtype, requires_grad):
    """is_causal=True: constant lower-triangular -inf bias (held fixed)."""
    out = []
    shapes = [(4, 6), (1, 1, 4, 6)]
    for i, s in enumerate(shapes):
        q = make_tensor(*s, dtype=dtype, requires_grad=requires_grad, seed=860 + i)
        k = make_tensor(*s, dtype=dtype, requires_grad=requires_grad, seed=870 + i)
        v = make_tensor(*s, dtype=dtype, requires_grad=requires_grad, seed=880 + i)
        out.append(SampleInput(q, k, v, is_causal=True))
    return out


# --------------------------------------------------------------------- op_db

# Resolve the op through nn.functional (the torch_compat-installed fallback) the same
# way core_ops binds F.layer_norm; a thin lambda keeps the binding eager and explicit.
def _sdpa_op(query, key, value, attn_mask=None, dropout_p=0.0,
             is_causal=False, scale=None):
    return F.scaled_dot_product_attention(
        query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
        is_causal=is_causal, scale=scale)


op_db = [
    # Smooth all the way through (matmul -> softmax -> matmul); full fwd + gradcheck
    # + gradgradcheck. The causal/explicit-scale knobs are constants held fixed under
    # differentiation, so they do not break smoothness.
    OpInfo("scaled_dot_product_attention",
           op=_sdpa_op, ref=scaled_dot_product_attention_ref,
           sample_inputs_func=sample_sdpa),
    OpInfo("scaled_dot_product_attention", variant_test_name="causal",
           op=_sdpa_op, ref=scaled_dot_product_attention_ref,
           sample_inputs_func=sample_sdpa_causal),
]
