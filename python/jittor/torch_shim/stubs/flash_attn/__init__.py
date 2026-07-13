"""A jittor-backed `flash_attn` so `import flash_attn` works on the torch-shim.

If a project provides a native ``flashattn_jittor`` implementation, this package
will discover/compile it through Jittor's torch-extension shim and dispatch the
standard flash-attn API to that module.  The fused implementation stays outside
the Jittor repository.  Without such a module, this shim provides the public
entry points TRELLIS.2 (and most torch libraries) call, computed by splitting the
packed (var-len) tensors per segment via the `cu_seqlens` prefix-sums and running
jittor's `torch.nn.functional.scaled_dot_product_attention` on each segment, then
re-packing. Block-diagonal attention with NO cross-segment leakage, i.e.
numerically equivalent to the real flash-attn var-len kernels (standard,
un-fused SDPA math).

Layouts (matching the real flash_attn API):
  * var-len q/k/v : [total_tokens, H, C]      (tokens packed across the batch)
    - packed qkv  : [total_tokens, 3, H, C]
    - packed  kv  : [total_tokens, 2, H, C]
    - cu_seqlens  : [B+1] int32 prefix-sums; segment i is [cu[i], cu[i+1])
    output        : [total_tokens, H, Cv]
  * dense q/k/v   : [B, L, H, C]              (fixed length L)
    - packed qkv  : [B, L, 3, H, C]
    - packed  kv  : [B, L, 2, H, C]
    output        : [B, L, H, Cv]

Deployed by `jittor.torch_shim.deploy` like the torchvision/torchaudio stubs,
and also registered into `sys.modules['flash_attn']` by the torch shim package
body so the `PYTHONPATH=.../python` (no-deploy) dev flow works too.
"""
import os
from typing import Optional

import torch
import torch.nn.functional as F

try:
    from jittor.torch_shim import flashattn_jittor as _flashattn_jittor
except Exception:  # pragma: no cover - fallback must remain import-safe
    _flashattn_jittor = None

__version__ = "2.7.4.post1"
_jittor_flash_attn_stub = True
_jittor_flash_attn_backend = "math"
_NATIVE_FUNCTION_CACHE = {}
_NATIVE_MISSING = object()
_TRUTHY = {"1", "true", "yes", "on"}

__all__ = [
    "flash_attn_func",
    "flash_attn_qkvpacked_func",
    "flash_attn_kvpacked_func",
    "flash_attn_varlen_func",
    "flash_attn_varlen_qkvpacked_func",
    "flash_attn_varlen_kvpacked_func",
    "flashattn_jittor_backend",
    "flashattn_jittor_last_error",
    "is_flashattn_jittor_available",
]


def _native_function(name):
    """Return a flashattn_jittor entry point, or None for math fallback."""
    global _jittor_flash_attn_backend
    if _flashattn_jittor is None:
        if _native_required():
            raise RuntimeError(
                "JITTOR_FLASH_ATTN_JITTOR_REQUIRED is set, but "
                "jittor.torch_shim.flashattn_jittor could not be imported"
            )
        _NATIVE_FUNCTION_CACHE[name] = (None, None)
        return None
    backend = _flashattn_jittor.load_backend()
    cached = _NATIVE_FUNCTION_CACHE.get(name, _NATIVE_MISSING)
    if cached is not _NATIVE_MISSING and cached[0] is backend:
        if backend is None and _flashattn_jittor.required():
            raise RuntimeError(
                "JITTOR_FLASH_ATTN_JITTOR_REQUIRED is set, but native "
                "flashattn_jittor is unavailable: %s"
                % (_flashattn_jittor.last_error() or "unknown error")
            )
        current = getattr(backend, name, None) if backend is not None else None
        if cached[1] is current:
            return cached[1]
    if backend is None:
        _jittor_flash_attn_backend = _flashattn_jittor.backend_name()
        if _flashattn_jittor.required():
            raise RuntimeError(
                "JITTOR_FLASH_ATTN_JITTOR_REQUIRED is set, but native "
                "flashattn_jittor is unavailable: %s"
                % (_flashattn_jittor.last_error() or "unknown error")
            )
        _NATIVE_FUNCTION_CACHE[name] = (None, None)
        return None
    _jittor_flash_attn_backend = _flashattn_jittor.backend_name()
    fn = getattr(backend, name, None)
    if callable(fn):
        _NATIVE_FUNCTION_CACHE[name] = (backend, fn)
        return fn
    if _flashattn_jittor.required():
        raise RuntimeError(
            "native flashattn_jittor backend %s does not provide %s"
            % (_jittor_flash_attn_backend, name)
        )
    _NATIVE_FUNCTION_CACHE[name] = (backend, None)
    return None


def _native_required():
    if _flashattn_jittor is not None:
        return bool(_flashattn_jittor.required())
    return any(
        str(os.environ.get(name) or "").strip().lower() in _TRUTHY
        for name in (
            "JITTOR_FLASH_ATTN_JITTOR_REQUIRED",
            "JITTOR_FLASHATTN_JITTOR_REQUIRED",
        )
    )


def _call_native(name, *args, **kwargs):
    fn = _native_function(name)
    if fn is None:
        return None
    out = fn(*args, **kwargs)
    if out is None and _native_required():
        raise RuntimeError(
            "native flashattn_jittor backend %s returned no output for %s"
            % (flashattn_jittor_backend(), name)
        )
    return out


def flashattn_jittor_backend():
    if _flashattn_jittor is None:
        return "math"
    return _flashattn_jittor.backend_name()


def flashattn_jittor_last_error():
    if _flashattn_jittor is None:
        return "jittor.torch_shim.flashattn_jittor could not be imported"
    return _flashattn_jittor.last_error()


def is_flashattn_jittor_available():
    return _flashattn_jittor is not None and _flashattn_jittor.is_available()


def _to_int_list(cu_seqlens):
    """cu_seqlens [B+1] (prefix-sums) -> python int segment-length list [B]."""
    if isinstance(cu_seqlens, torch.Tensor):
        vals = [int(x) for x in cu_seqlens.reshape(-1).tolist()]
    else:
        vals = [int(x) for x in cu_seqlens]
    return [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]


def _sdpa_segment(q, k, v, causal, softmax_scale):
    """q [Lq,H,Cq], k [Lk,H,C], v [Lk,H,Cv] -> [Lq,H,Cv] via one SDPA call.

    Treats the single segment as a batch of 1; SDPA wants [B,H,L,E]."""
    if q.shape[0] == 0:
        return torch.zeros((0, q.shape[1], v.shape[2]), dtype=q.dtype, device=q.device)
    qh = q.permute(1, 0, 2).unsqueeze(0)   # [1,H,Lq,Cq]
    kh = k.permute(1, 0, 2).unsqueeze(0)   # [1,H,Lk,C]
    vh = v.permute(1, 0, 2).unsqueeze(0)   # [1,H,Lk,Cv]
    oh = F.scaled_dot_product_attention(
        qh, kh, vh, is_causal=bool(causal), scale=softmax_scale)
    return oh.squeeze(0).permute(1, 0, 2)  # [Lq,H,Cv]


def _varlen_core(q, k, v, cu_seqlens_q, cu_seqlens_k, causal, softmax_scale):
    """Per-segment SDPA over packed [total,H,C] var-len tensors -> [total,H,Cv]."""
    q_seqlen = _to_int_list(cu_seqlens_q)
    kv_seqlen = _to_int_list(cu_seqlens_k)
    assert len(q_seqlen) == len(kv_seqlen), \
        f"cu_seqlens batch mismatch: {len(q_seqlen)} vs {len(kv_seqlen)}"
    outs = []
    qs = ks = 0
    for ql, kl in zip(q_seqlen, kv_seqlen):
        outs.append(_sdpa_segment(
            q[qs:qs + ql], k[ks:ks + kl], v[ks:ks + kl], causal, softmax_scale))
        qs += ql
        ks += kl
    if not outs:
        return torch.zeros((0, q.shape[1], v.shape[2]), dtype=q.dtype, device=q.device)
    return torch.cat(outs, dim=0)


# ---------------------------------------------------------------- var-len API

def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k,
                           max_seqlen_q, max_seqlen_k,
                           dropout_p=0.0, softmax_scale=None, causal=False,
                           *args, **kwargs):
    """q,k,v: [total, H, C]; cu_seqlens_*: [B+1] int32. -> [total, H, Cv]."""
    out = _call_native(
        "flash_attn_varlen_func", q, k, v, cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k, dropout_p, softmax_scale, causal,
        *args, **kwargs)
    if out is not None:
        return out
    return _varlen_core(q, k, v, cu_seqlens_q, cu_seqlens_k, causal, softmax_scale)


def flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens, max_seqlen,
                                     dropout_p=0.0, softmax_scale=None,
                                     causal=False, *args, **kwargs):
    """qkv: [total, 3, H, C]; cu_seqlens: [B+1] int32. -> [total, H, C]."""
    out = _call_native(
        "flash_attn_varlen_qkvpacked_func", qkv, cu_seqlens, max_seqlen,
        dropout_p, softmax_scale, causal, *args, **kwargs)
    if out is not None:
        return out
    q = qkv[:, 0]
    k = qkv[:, 1]
    v = qkv[:, 2]
    return _varlen_core(q, k, v, cu_seqlens, cu_seqlens, causal, softmax_scale)


def flash_attn_varlen_kvpacked_func(q, kv, cu_seqlens_q, cu_seqlens_k,
                                    max_seqlen_q, max_seqlen_k,
                                    dropout_p=0.0, softmax_scale=None,
                                    causal=False, *args, **kwargs):
    """q: [total_q, H, C]; kv: [total_kv, 2, H, C]. -> [total_q, H, Cv]."""
    out = _call_native(
        "flash_attn_varlen_kvpacked_func", q, kv, cu_seqlens_q,
        cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p,
        softmax_scale, causal, *args, **kwargs)
    if out is not None:
        return out
    k = kv[:, 0]
    v = kv[:, 1]
    return _varlen_core(q, k, v, cu_seqlens_q, cu_seqlens_k, causal, softmax_scale)


# ------------------------------------------------------------------ dense API

def _dense_core(q, k, v, causal, softmax_scale):
    """q,k,v: [B, L*, H, C] -> [B, Lq, H, Cv] via batched SDPA."""
    qh = q.permute(0, 2, 1, 3)             # [B,H,Lq,Cq]
    kh = k.permute(0, 2, 1, 3)             # [B,H,Lk,C]
    vh = v.permute(0, 2, 1, 3)             # [B,H,Lk,Cv]
    oh = F.scaled_dot_product_attention(
        qh, kh, vh, is_causal=bool(causal), scale=softmax_scale)
    return oh.permute(0, 2, 1, 3)          # [B,Lq,H,Cv]


def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False,
                    *args, **kwargs):
    """q,k,v: [B, L, H, C]. -> [B, L, H, Cv]."""
    out = _call_native(
        "flash_attn_func", q, k, v, dropout_p, softmax_scale, causal,
        *args, **kwargs)
    if out is not None:
        return out
    return _dense_core(q, k, v, causal, softmax_scale)


def flash_attn_qkvpacked_func(qkv, dropout_p=0.0, softmax_scale=None,
                              causal=False, *args, **kwargs):
    """qkv: [B, L, 3, H, C]. -> [B, L, H, C]."""
    out = _call_native(
        "flash_attn_qkvpacked_func", qkv, dropout_p, softmax_scale,
        causal, *args, **kwargs)
    if out is not None:
        return out
    q = qkv[:, :, 0]
    k = qkv[:, :, 1]
    v = qkv[:, :, 2]
    return _dense_core(q, k, v, causal, softmax_scale)


def flash_attn_kvpacked_func(q, kv, dropout_p=0.0, softmax_scale=None,
                             causal=False, *args, **kwargs):
    """q: [B, Lq, H, C]; kv: [B, Lkv, 2, H, C]. -> [B, Lq, H, Cv]."""
    out = _call_native(
        "flash_attn_kvpacked_func", q, kv, dropout_p, softmax_scale,
        causal, *args, **kwargs)
    if out is not None:
        return out
    k = kv[:, :, 0]
    v = kv[:, :, 1]
    return _dense_core(q, k, v, causal, softmax_scale)
