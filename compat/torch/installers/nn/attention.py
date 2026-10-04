from jittor._core.dtypes import dtype_name as _jittor_dtype_name
import jittor as jt
from jittor import nn
from jittor.nn.backends import hooks as _backend_hooks
from ....diagnostics import EXPECTED, swallowed, sdpa_flash_stats

from ...context import get_install_context
from ...fidelity import Fidelity, register_fidelity

from jittor._runtime.environment import getenv as _getenv


def _sdpa_static_backend_cache_enabled():
    return (_getenv("JITTOR_TORCH_INFERENCE") or "").strip().lower() \
        in ("1", "true", "yes", "on")


def _sdpa_flash_stats():
    return sdpa_flash_stats(jt)


def _sdpa_flash_miss(reason):
    misses = _sdpa_flash_stats()["misses"]
    misses[reason] = misses.get(reason, 0) + 1
    _LAST_MISS[0] = reason


_LAST_MISS = [None]


def _sdpa_flash_cast(reason):
    casts = _sdpa_flash_stats()["casts"]
    casts[reason] = casts.get(reason, 0) + 1


def _sdpa_flash_hit(backend_name):
    stats = _sdpa_flash_stats()
    stats["hits"] += 1
    stats["backend"] = backend_name


def _sdpa_flash_template_dim(dim):
    dim = int(dim)
    if dim <= 0 or dim > 256 or dim % 8 != 0:
        return None
    if dim <= 32:
        return 32
    if dim <= 64:
        return 64
    if dim <= 96:
        return 96
    if dim <= 128:
        return 128
    if dim <= 192:
        return 192
    return 256


def _sdpa_flash_float32_cast_target():
    raw = (_getenv("JITTOR_FLASH_ATTN_CAST_FLOAT32") or "").strip().lower()
    if raw in ("1", "true", "yes", "on", "fp16", "float16", "half"):
        return "float16"
    if raw in ("bf16", "bfloat16"):
        return "bfloat16"
    return None


def _try_flash_scaled_dot_product_attention(query, key, value, attn_mask,
                                            dropout_p, is_causal, sf,
                                            enable_gqa=False):
    _sdpa_flash_backend_cache = get_install_context(jt).state["sdpa_backend_cache"]
    acl_attention = _backend_hooks.acl_scaled_dot_product_attention
    # The hook answers with whatever kernel the backend registered. On CUDA
    # that is the native flash kernel, which loads the same flash library the
    # path below does but without its gates -- the capability-checked backend
    # cache, compact GQA heads, the statistics -- so the path below takes it.
    if callable(acl_attention) and \
            not getattr(acl_attention, "torch_frontend_loads_directly", False):
        acl_output = acl_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=sf,
            enable_gqa=enable_gqa,
        )
        if acl_output is not None:
            _sdpa_flash_hit(getattr(
                acl_attention, "backend_name",
                "acl_flash_attention_score_v2"))
            return acl_output
    if attn_mask is not None:
        _sdpa_flash_miss("mask")
        return None
    dropout = float(dropout_p or 0.0)
    if dropout < 0.0 or dropout >= 1.0:
        _sdpa_flash_miss("dropout_range")
        return None
    if not jt.flags.use_cuda:
        _sdpa_flash_miss("not_cuda")
        return None
    training_requested = not getattr(jt.flags, "no_grad", 0) or dropout != 0.0
    q_shape, k_shape, v_shape = tuple(query.shape), tuple(key.shape), tuple(value.shape)
    if len(q_shape) < 3 or len(q_shape) != len(k_shape) or len(q_shape) != len(v_shape):
        _sdpa_flash_miss("rank")
        return None
    if q_shape[:-3] != k_shape[:-3] or q_shape[:-3] != v_shape[:-3]:
        _sdpa_flash_miss("batch")
        return None
    query_heads = int(q_shape[-3])
    key_heads = int(k_shape[-3])
    value_heads = int(v_shape[-3])
    gqa_heads_ok = (key_heads > 0 and enable_gqa
                    and query_heads % key_heads == 0)
    if key_heads != value_heads or not (
            query_heads == key_heads or gqa_heads_ok):
        _sdpa_flash_miss("heads")
        return None
    if q_shape[-1] != k_shape[-1] or q_shape[-1] != v_shape[-1]:
        _sdpa_flash_miss("head_dim_mismatch")
        return None
    # For CLIP-style short self-attention, the two cuBLAS matmuls plus the
    # fused softmax are faster than materializing the three layout copies
    # required by the separate-QKV FlashAttention wrapper. Keep this
    # inference-only and narrowly shaped so decoding, GQA and training keep
    # their existing backend choice.
    # The environment last: every other test is cheaper, and almost no call
    # passes them.
    short_square_math = (
        not enable_gqa and not is_causal
        and len(q_shape) == 4 and 0 < int(q_shape[0]) <= 8
        and query_heads == key_heads == value_heads == 12
        and int(q_shape[-1]) == 64
        and int(q_shape[-2]) == int(k_shape[-2]) == int(v_shape[-2])
        and int(q_shape[-2]) <= 64
        and _jittor_dtype_name(query.dtype) == _jittor_dtype_name(key.dtype) == _jittor_dtype_name(value.dtype)
        and _jittor_dtype_name(query.dtype) == "float16"
        and _sdpa_static_backend_cache_enabled())
    if short_square_math:
        _sdpa_flash_miss("short_square_math")
        return None
    template_dim = _sdpa_flash_template_dim(q_shape[-1])
    if template_dim is None:
        _sdpa_flash_miss("head_dim")
        return None
    q_dtype, k_dtype, v_dtype = _jittor_dtype_name(query.dtype), _jittor_dtype_name(key.dtype), _jittor_dtype_name(value.dtype)
    original_dtype = q_dtype
    cast_back = False
    if not (q_dtype == k_dtype == v_dtype and _jittor_dtype_name(q_dtype) in ("float16", "bfloat16")):
        cast_target = _sdpa_flash_float32_cast_target()
        if cast_target is None or not (_jittor_dtype_name(q_dtype) == _jittor_dtype_name(k_dtype) == _jittor_dtype_name(v_dtype) == "float32"):
            _sdpa_flash_miss("dtype")
            return None
        query = query.to(cast_target)
        key = key.to(cast_target)
        value = value.to(cast_target)
        q_dtype = k_dtype = v_dtype = cast_target
        cast_back = True
        _sdpa_flash_cast("float32_to_%s" % cast_target)
    try:
        from jittor.compat.shim.backends import flash_attention as _fa_jittor
    except EXPECTED as exc:
        swallowed("torch/installers/nn.py _try_flash_scaled_dot_product_attention: from jittor.compat.shim.backends import flash_attention...", exc)
        _sdpa_flash_miss("no_loader")
        return None
    # Nothing to load: the miss everything below would end in, without the
    # environment reads on the way. A training call keeps the full walk, which
    # can decline it for the score count first, and so does the static
    # backend cache, which keeps its own per-token answers.
    if (not training_requested and not _sdpa_static_backend_cache_enabled()
            and _fa_jittor.known_unavailable(template_dim, q_dtype)
            and not _fa_jittor.required()):
        _sdpa_flash_miss("no_backend")
        return None
    if training_requested and dropout == 0.0 and not _fa_jittor.required():
        min_scores = _fa_jittor.training_min_scores()
        score_elements = query_heads * int(q_shape[-2]) * int(k_shape[-2])
        for size in q_shape[:-3]:
            score_elements *= int(size)
        if min_scores and score_elements < min_scores:
            _sdpa_flash_miss("short_training_math")
            return None
    cache_key = (template_dim, q_dtype)
    static_cache = _sdpa_static_backend_cache_enabled() or training_requested
    token_fn = getattr(_fa_jittor, "backend_cache_token", None)
    backend_token = (token_fn() if static_cache and callable(token_fn)
                     else None)
    cached = (_sdpa_flash_backend_cache.get(cache_key)
              if static_cache and backend_token is not None else None)
    if cached is not None and cached[0] == backend_token:
        backend, capability_miss = cached[1], None
    else:
        backend, capability_miss = _fa_jittor.load_backend_for(
            template_dim, q_dtype)
        publication_fn = getattr(
            _fa_jittor, "backend_publication_token", None)
        publication_token = (
            publication_fn(backend) if callable(publication_fn) else None)
        backend_token = (token_fn() if static_cache and callable(token_fn)
                         else None)
        if (static_cache and backend_token is not None
                and publication_token == backend_token
                and backend is not None and capability_miss is None):
            _sdpa_flash_backend_cache[cache_key] = (
                backend_token,
                backend,
            )
    if backend is None:
        if _fa_jittor.required():
            raise RuntimeError(
                "JITTOR_FLASH_ATTN_JITTOR_REQUIRED is set, but native "
                "flash-attn backend is unavailable: %s"
                % (_fa_jittor.last_error() or "unknown error")
            )
        _sdpa_flash_miss("no_backend")
        return None
    if capability_miss is not None:
        if _fa_jittor.required():
            raise RuntimeError(
                "native flash-attn backend could not expand for %s: %s"
                % (capability_miss, _fa_jittor.last_error() or "unsupported capability")
            )
        _sdpa_flash_miss(capability_miss)
        return None
    if (training_requested
            and not getattr(backend, "_flashattn_jittor_training", False)):
        if _fa_jittor.required():
            raise RuntimeError(
                "native flash-attn backend does not advertise backward/dropout support")
        _sdpa_flash_miss("no_training_backend")
        return None
    # load_backend_for() already returned the capability-checked backend.
    # Calling through the public flash_attn stub would invoke the loader a
    # second time for every layer and rescan all backend environment keys.
    fn = getattr(backend, "flash_attn_func", None)
    if not callable(fn):
        if _fa_jittor.required():
            raise RuntimeError("flash_attn shim has no flash_attn_func")
        _sdpa_flash_miss("no_func")
        return None
    prefix = q_shape[:-3]
    p = len(prefix)
    batch = 1
    for size in prefix:
        batch *= int(size)
    heads, lq, head_dim = query_heads, int(q_shape[-2]), int(q_shape[-1])
    lk = int(k_shape[-2])
    q_axes = tuple(list(range(p)) + [p + 1, p, p + 2])
    # Native flash-attn is an external C++/CUDA extension. Crossing that
    # boundary with a lazy permute/reshape expression can leave the bridge
    # holding transient metadata; clone materializes a stable row-major
    # tensor while keeping the kernel path fused.
    q_dense = query.permute(*q_axes).reshape((batch, lq, heads, head_dim)).clone()
    k_dense = key.permute(*q_axes).reshape((batch, lk, key_heads, head_dim)).clone()
    v_dense = value.permute(*q_axes).reshape((batch, lk, value_heads, head_dim)).clone()
    try:
        out = fn(
            q_dense, k_dense, v_dense, dropout, float(sf), bool(is_causal))
    except EXPECTED as exc:
        swallowed("torch/installers/nn.py _try_flash_scaled_dot_product_attention: out = fn(", exc)
        if _fa_jittor.required():
            raise
        _sdpa_flash_miss("call_failed")
        return None
    if out is None:
        if _fa_jittor.required():
            raise RuntimeError(
                "native flash-attn backend returned no output while "
                "JITTOR_FLASH_ATTN_JITTOR_REQUIRED is set"
            )
        _sdpa_flash_miss("returned_none")
        return None
    out = out.reshape(tuple(prefix) + (lq, heads, head_dim))
    out_axes = tuple(list(range(p)) + [p + 1, p, p + 2])
    _sdpa_flash_hit(_fa_jittor.backend_name())
    out = out.permute(*out_axes)
    if cast_back and _jittor_dtype_name(out.dtype) != original_dtype:
        out = out.to(original_dtype)
    return out


import math as _math


from jittor.nn.functional.attention import (
    scaled_dot_product_attention as _native_scaled_dot_product_attention,
    LAST_FUSED_KERNEL as _NATIVE_LAST_FUSED,
)


#: Inference calls already walked: their signature -> (the flash miss, the
#: fused kernel that answered). See `_route_key`.
_ROUTES = {}


def _route_key(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa):
    """What decides where an inference call goes, or None if it is not one
    to remember.

    The walk -- the flash attempt here, the native function's flash kernel,
    then `nn.fused_attention`'s selection -- reads only these: shapes, dtypes,
    the mask's shape and dtype, the switches, grad mode, the kernel registry
    and the flash loader's token. 17-27 us of Python a call, where the kernel
    it ends in builds its operator in a third of that. Whether key and value
    are grouped heads still in the making (`_repeated_heads`) depends on how
    they were built, so a key whose producer is a reshape is not remembered.
    """
    if not jt.flags.no_grad or not jt.flags.use_cuda or jt.flags.amp_reg \
            or enable_gqa or dropout_p or not isinstance(query, jt.Var) \
            or not isinstance(key, jt.Var) or not isinstance(value, jt.Var):
        return None
    if scale is not None and type(scale) not in (int, float):
        return None
    if key._producer_name() == "reshape" or value._producer_name() == "reshape":
        return None
    if attn_mask is None:
        mask = None
    elif isinstance(attn_mask, jt.Var):
        mask = (tuple(attn_mask.shape), str(attn_mask.dtype))
    else:
        return None
    # The static backend cache keeps its own answers per token; leave its
    # calls to it.
    if _sdpa_static_backend_cache_enabled():
        return None
    loader = _flash_loader[0]
    if loader is None:
        from jittor.compat.shim.backends import flash_attention as _fa_jittor
        loader = _flash_loader[0] = _fa_jittor
    token = loader.backend_cache_token()
    if token is None:
        return None
    return (tuple(query.shape), tuple(key.shape), tuple(value.shape),
            str(query.dtype), str(key.dtype), str(value.dtype), mask, bool(is_causal),
            None if scale is None else float(scale), jt.flags.device_id,
            _dispatch_state[0].generation, token)


_flash_loader = [None]
from jittor._runtime import dispatch as _dispatch_module
_dispatch_state = [_dispatch_module]


def scaled_dot_product_attention(query, key, value, attn_mask=None,
                                 dropout_p=0.0, is_causal=False,
                                 scale=None, enable_gqa=False, **kw):
    del kw
    route_key = _route_key(query, key, value, attn_mask, dropout_p, is_causal,
                           scale, enable_gqa)
    if route_key is not None:
        route = _ROUTES.get(route_key)
        if route is not None:
            miss, kernel = route
            out = kernel(query, key, value, attn_mask=attn_mask, dropout_p=0.0,
                         is_causal=is_causal, scale=scale)
            if out is not None:
                _sdpa_flash_miss(miss)
                return out
            del _ROUTES[route_key]
    dimension = int(query.shape[-1])
    scale_factor = (
        1.0 / _math.sqrt(dimension) if scale is None else scale
    )
    _LAST_MISS[0] = None
    flash = _try_flash_scaled_dot_product_attention(
        query, key, value, attn_mask, dropout_p, is_causal,
        scale_factor, enable_gqa=enable_gqa)
    if flash is not None:
        return flash
    miss = _LAST_MISS[0]
    _NATIVE_LAST_FUSED[0] = None
    out = _native_scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        enable_gqa=enable_gqa,
    )
    # Remember the walk only where it ended in a fused kernel and flash had
    # declined for a reason the key holds: a mask, or nothing to load.
    kernel = _NATIVE_LAST_FUSED[0]
    if route_key is not None and kernel is not None and miss in ("mask", "no_backend") \
            and len(_ROUTES) < 256:
        _ROUTES[route_key] = (miss, kernel)
    return out


register_fidelity("torch.nn.functional.scaled_dot_product_attention", scaled_dot_product_attention, Fidelity.APPROXIMATE, "native attention mathematics with backend selection and GQA expansion; backend capability and precision restrictions apply")

def install_attention(ctx):
    nn = ctx.target_namespace.nn
    g = ctx.jittor_module

    ctx.state["sdpa_backend_cache"] = {}









    # The Torch wrapper owns backend selection and GQA expansion. The math
    # fallback remains the canonical native functional implementation.


    nn.functional.scaled_dot_product_attention = scaled_dot_product_attention
    g.scaled_dot_product_attention = nn.functional.scaled_dot_product_attention
    g._torch_sdpa_flash_backend_cache = ctx.state["sdpa_backend_cache"]
