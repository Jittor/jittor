"""Runtime patches for running TRELLIS.2 through the Jittor torch shim.

These patches live in Jittor rather than in TRELLIS.2 so the upstream project
and its native extension dependencies can stay unmodified.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import math
import os
import sys
from typing import Dict, Tuple


_FALSEY = {"0", "false", "no", "off"}
_CU_SEQLENS_CACHE: Dict[Tuple[str, Tuple[int, ...], str], object] = {}
_DENSE_ATTENTION_MODULE = "trellis2.modules.attention.full_attn"
_ATTENTION_MODULE = "trellis2.modules.attention.modules"
_SPARSE_ATTENTION_MODULE = "trellis2.modules.sparse.attention.full_attn"
_DINOV3_MODULE = "transformers.models.dinov3_vit.modeling_dinov3_vit"
_LAYOUT_INFO_KEY = "_jittor_torch_layout_info"
_CU_SEQLENS_KEY = "_jittor_torch_cu_seqlens_int32"


def _is_falsey(value) -> bool:
    return str(value or "").strip().lower() in _FALSEY


def _cached_cu_seqlens(torch, lengths, device):
    key = (str(device), tuple(int(x) for x in lengths), "int32")
    out = _CU_SEQLENS_CACHE.get(key)
    if out is None:
        values = [0]
        total = 0
        for length in key[1]:
            total += int(length)
            values.append(total)
        out = torch.tensor(values, dtype=torch.int32, device=device)
        _CU_SEQLENS_CACHE[key] = out
    return out


def _cache_get(obj, key):
    cache = getattr(obj, "_cache", None)
    if isinstance(cache, dict):
        return cache.get(key)
    getter = getattr(obj, "get_spatial_cache", None)
    if getter is not None:
        try:
            return getter(key)
        except Exception:
            return None
    return None


def _cache_set(obj, key, value) -> None:
    cache = getattr(obj, "_cache", None)
    if isinstance(cache, dict):
        cache[key] = value
        return
    setter = getattr(obj, "register_spatial_cache", None)
    if setter is not None:
        try:
            setter(key, value)
        except Exception:
            pass


def _layout_info(obj):
    info = _cache_get(obj, _LAYOUT_INFO_KEY)
    if info is not None:
        return info
    lengths = tuple(item.stop - item.start for item in obj.layout)
    max_len = max(lengths) if lengths else 0
    info = (lengths, max_len)
    _cache_set(obj, _LAYOUT_INFO_KEY, info)
    return info


def _cached_varlen_cu_seqlens(torch, obj):
    lengths, max_len = _layout_info(obj)
    device = obj.device
    device_key = str(device)
    cached = _cache_get(obj, _CU_SEQLENS_KEY)
    if cached is not None and cached[0] == device_key:
        return lengths, max_len, cached[1]
    cu = _cached_cu_seqlens(torch, lengths, device)
    _cache_set(obj, _CU_SEQLENS_KEY, (device_key, cu))
    return lengths, max_len, cu


def _dense_lengths(n, length):
    length = int(length)
    return (length,) * int(n), length


def _flash_funcs(mod, names):
    funcs = mod.__dict__.get("_jittor_torch_flash_attn_funcs")
    if funcs is None:
        import flash_attn

        funcs = tuple(getattr(flash_attn, name) for name in names)
        mod.__dict__["_jittor_torch_flash_attn_funcs"] = funcs
    return funcs


def _patch_dense_attention_module(mod) -> bool:
    if getattr(mod, "_jittor_torch_fast_dense_attn", False):
        return True

    try:
        import importlib

        torch = importlib.import_module("torch")
        from trellis2.modules.attention import config
    except Exception:
        return False

    original = getattr(mod, "scaled_dot_product_attention", None)
    if original is None:
        return False

    def scaled_dot_product_attention(*args, **kwargs):
        if config.BACKEND != "flash_attn":
            return original(*args, **kwargs)
        qkvpacked_func, kvpacked_func, attn_func = _flash_funcs(
            mod,
            ("flash_attn_qkvpacked_func", "flash_attn_kvpacked_func", "flash_attn_func"),
        )
        if not kwargs:
            if len(args) == 1:
                return qkvpacked_func(args[0])
            if len(args) == 2:
                return kvpacked_func(args[0], args[1])
            if len(args) == 3:
                return attn_func(args[0], args[1], args[2])
            return original(*args, **kwargs)

        num_all_args = len(args) + len(kwargs)
        if num_all_args == 1 and (args or "qkv" in kwargs):
            return qkvpacked_func(args[0] if args else kwargs["qkv"])
        if num_all_args == 2 and all(key in kwargs for key in ("q", "kv")[len(args):]):
            q = args[0] if len(args) > 0 else kwargs["q"]
            kv = args[1] if len(args) > 1 else kwargs["kv"]
            return kvpacked_func(q, kv)
        if num_all_args == 3 and all(key in kwargs for key in ("q", "k", "v")[len(args):]):
            q = args[0] if len(args) > 0 else kwargs["q"]
            k = args[1] if len(args) > 1 else kwargs["k"]
            v = args[2] if len(args) > 2 else kwargs["v"]
            return attn_func(q, k, v)
        return original(*args, **kwargs)

    scaled_dot_product_attention._jittor_torch_fast_dense_attn = True
    scaled_dot_product_attention._jittor_torch_original = original
    mod.scaled_dot_product_attention = scaled_dot_product_attention

    refs = sys.modules.get("trellis2.modules.attention.modules")
    if refs is not None:
        try:
            refs.scaled_dot_product_attention = scaled_dot_product_attention
        except Exception:
            pass
    return True


def _patch_sparse_attention_module(mod) -> bool:
    if getattr(mod, "_jittor_torch_fast_sparse_attn", False):
        return True

    try:
        import importlib

        torch = importlib.import_module("torch")
        from trellis2.modules.sparse import config
        from trellis2.modules.sparse.basic import VarLenTensor
    except Exception:
        return False

    original = getattr(mod, "sparse_scaled_dot_product_attention", None)
    if original is None:
        return False

    def sparse_scaled_dot_product_attention(*args, **kwargs):
        if config.ATTN != "flash_attn":
            return original(*args, **kwargs)
        if kwargs:
            return original(*args, **kwargs)
        varlen_func, qkvpacked_func, kvpacked_func = _flash_funcs(
            mod,
            (
                "flash_attn_varlen_func",
                "flash_attn_varlen_qkvpacked_func",
                "flash_attn_varlen_kvpacked_func",
            ),
        )
        num_all_args = len(args)
        if num_all_args not in (1, 2, 3):
            return original(*args, **kwargs)

        if num_all_args == 1:
            qkv = args[0]
            if not isinstance(qkv, VarLenTensor):
                return original(*args, **kwargs)
            _, max_q, cu_q = _cached_varlen_cu_seqlens(torch, qkv)
            out = qkvpacked_func(qkv.feats, cu_q, max_q)
            return qkv.replace(out)

        if num_all_args == 2:
            q, kv = args
            if not (
                isinstance(q, VarLenTensor)
                and isinstance(kv, (VarLenTensor, torch.Tensor))
                or isinstance(q, torch.Tensor)
                and isinstance(kv, VarLenTensor)
            ):
                return original(*args, **kwargs)
            if isinstance(q, VarLenTensor):
                s = q
                q_seqlen, max_q, cu_q = _cached_varlen_cu_seqlens(torch, q)
                q_feats = q.feats
            else:
                s = None
                n, l, h, c = q.shape
                q_seqlen, max_q = _dense_lengths(n, l)
                q_feats = q.reshape(n * l, h, c)
                cu_q = _cached_cu_seqlens(torch, q_seqlen, q.device)
            if isinstance(kv, VarLenTensor):
                kv_seqlen, max_kv, cu_kv = _cached_varlen_cu_seqlens(torch, kv)
                kv_feats = kv.feats
            else:
                n, l, _, h, c = kv.shape
                kv_seqlen, max_kv = _dense_lengths(n, l)
                kv_feats = kv.reshape(n * l, 2, h, c)
                cu_kv = _cached_cu_seqlens(torch, kv_seqlen, q.device)
            if q_seqlen == kv_seqlen:
                cu_kv = cu_q
            out = kvpacked_func(q_feats, kv_feats, cu_q, cu_kv, max_q, max_kv)
            return s.replace(out) if s is not None else out.reshape(n, l, h, -1)

        q, k, v = args
        if isinstance(q, VarLenTensor):
            s = q
            q_seqlen, max_q, cu_q = _cached_varlen_cu_seqlens(torch, q)
            q_feats = q.feats
        else:
            s = None
            n, l, h, ci = q.shape
            q_seqlen, max_q = _dense_lengths(n, l)
            q_feats = q.reshape(n * l, h, ci)
            cu_q = _cached_cu_seqlens(torch, q_seqlen, q.device)
        if isinstance(k, VarLenTensor):
            if not isinstance(v, VarLenTensor):
                return original(*args, **kwargs)
            kv_seqlen, max_kv, cu_kv = _cached_varlen_cu_seqlens(torch, k)
            k_feats = k.feats
            v_feats = v.feats
        else:
            n, l, h, ci = k.shape
            kv_seqlen, max_kv = _dense_lengths(n, l)
            k_feats = k.reshape(n * l, h, ci)
            v_feats = v.reshape(n * l, h, v.shape[-1])
            cu_kv = _cached_cu_seqlens(torch, kv_seqlen, q.device)
        if q_seqlen == kv_seqlen:
            cu_kv = cu_q
        out = varlen_func(q_feats, k_feats, v_feats, cu_q, cu_kv, max_q, max_kv)
        return s.replace(out) if s is not None else out.reshape(n, l, h, -1)

    sparse_scaled_dot_product_attention._jittor_torch_fast_sparse_attn = True
    sparse_scaled_dot_product_attention._jittor_torch_original = original
    mod.sparse_scaled_dot_product_attention = sparse_scaled_dot_product_attention

    refs = sys.modules.get("trellis2.modules.sparse.attention.modules")
    if refs is not None:
        try:
            refs.sparse_scaled_dot_product_attention = sparse_scaled_dot_product_attention
        except Exception:
            pass
    return True


def _trellis_multihead_rms_norm_fast_path(x, gamma, scale):
    if _is_falsey(os.environ.get("JITTOR_TRELLIS_FUSED_RMS_NORM")):
        return None
    try:
        import jittor as jt
    except Exception:
        return None

    if not isinstance(x, jt.Var) or not isinstance(gamma, jt.Var):
        return None
    if not (jt.flags.use_cuda and getattr(jt.flags, "no_grad", 0)):
        return None
    if getattr(jt.compiler, "has_acl", 0):
        return None
    if str(x.dtype) != "bfloat16" or str(gamma.dtype) != "float32":
        return None
    try:
        devices = (int(x.get_device()), int(gamma.get_device()))
        if any(index < 0 for index in devices) or devices[0] != devices[1]:
            return None
        scale_value = float(scale)
    except Exception:
        return None

    x_shape = tuple(int(size) for size in x.shape)
    gamma_shape = tuple(int(size) for size in gamma.shape)
    if len(x_shape) != 4 or x_shape[-2:] != (12, 128):
        return None
    if x_shape[0] <= 0 or x_shape[1] <= 0 or gamma_shape != (12, 128):
        return None
    if (not math.isfinite(scale_value)
            or abs(scale_value - math.sqrt(128.0)) > 1e-6):
        return None

    cuda_src = r"""
    __device__ __forceinline__ float warp_sum(float value) {
        for (int offset = 16; offset > 0; offset >>= 1)
            value += __shfl_down_sync(0xffffffff, value, offset);
        return value;
    }
    __global__ static void jt_trellis_multihead_rms_norm_bf16(
            const in0_type* x, const in1_type* gamma, out0_type* y) {
        int row = blockIdx.x;
        int dim = threadIdx.x;
        int lane = dim & 31;
        int warp = dim >> 5;
        __shared__ float warp_buf[4];
        __shared__ float denominator;
        float value = static_cast<float>(x[row * 128 + dim]);
        float sum = warp_sum(value * value);
        if (lane == 0) warp_buf[warp] = sum;
        __syncthreads();
        if (warp == 0) {
            float total = lane < 4 ? warp_buf[lane] : 0.0f;
            total = warp_sum(total);
            if (lane == 0)
                denominator = fmaxf(sqrtf(total), 1.0e-12f);
        }
        __syncthreads();
        int head = row % 12;
        float weight = static_cast<float>(gamma[head * 128 + dim]);
        y[row * 128 + dim] = out0_type(
            value / denominator * weight * 11.313708498984761f);
    }
    int rows = in0->num / 128;
    jt_trellis_multihead_rms_norm_bf16<<<rows, 128>>>(
        in0_p, in1_p, out0_p);
    CHECK(0 == cudaGetLastError());
    """
    return jt.code(x.shape, x.dtype, [x, gamma], cuda_src=cuda_src)


def _patch_attention_module(mod) -> bool:
    cls = getattr(mod, "MultiHeadRMSNorm", None)
    original = getattr(cls, "forward", None) if cls is not None else None
    if original is None:
        return False
    if getattr(original, "_jittor_torch_fast_trellis_rms_norm", False):
        mod._jittor_torch_fast_trellis_rms_norm = True
        return True

    def forward(self, x, *args, **kwargs):
        if not args and not kwargs:
            fast = _trellis_multihead_rms_norm_fast_path(
                x, getattr(self, "gamma", None), getattr(self, "scale", None))
            if fast is not None:
                return fast
        return original(self, x, *args, **kwargs)

    forward._jittor_torch_fast_trellis_rms_norm = True
    forward._jittor_torch_original = original
    cls.forward = forward
    mod._jittor_torch_fast_trellis_rms_norm = True
    return True


def _dinov3_rotary_fast_path(q, k, cos, sin):
    if _is_falsey(os.environ.get("JITTOR_DINOV3_FUSED_ROPE")):
        return None
    try:
        import jittor as jt
    except Exception:
        return None

    tensors = (q, k, cos, sin)
    if not all(isinstance(value, jt.Var) for value in tensors):
        return None
    if not (jt.flags.use_cuda and getattr(jt.flags, "no_grad", 0)):
        return None
    if getattr(jt.compiler, "has_acl", 0):
        return None
    if not all(str(value.dtype) == "float32" for value in tensors):
        return None
    try:
        devices = tuple(int(value.get_device()) for value in tensors)
        if any(index < 0 for index in devices) or len(set(devices)) != 1:
            return None
    except Exception:
        return None

    q_shape = tuple(int(size) for size in q.shape)
    k_shape = tuple(int(size) for size in k.shape)
    cos_shape = tuple(int(size) for size in cos.shape)
    sin_shape = tuple(int(size) for size in sin.shape)
    if q_shape != k_shape or len(q_shape) != 4:
        return None
    if q_shape[0] <= 0 or q_shape[1] != 16 or q_shape[-1] != 64:
        return None
    if cos_shape != sin_shape or len(cos_shape) != 2:
        return None
    patch_count, head_dim = cos_shape
    token_count = q_shape[-2]
    if head_dim != 64 or patch_count <= 0 or patch_count > token_count:
        return None
    if int(cos.numel()) != patch_count * head_dim:
        return None

    cuda_src = r"""
    __global__ static void jt_dinov3_rope_fp32(
            const float* q, const float* k,
            const float* cos, const float* sin,
            float* out_q, float* out_k,
            int token_count, int patch_count) {
        int row = blockIdx.x;
        int dim = threadIdx.x;
        int index = row * 64 + dim;
        int token = row % token_count;
        int prefix_count = token_count - patch_count;
        if (token < prefix_count) {
            out_q[index] = q[index];
            out_k[index] = k[index];
            return;
        }

        int position = token - prefix_count;
        int other_dim = dim ^ 32;
        int other_index = row * 64 + other_dim;
        float sign = dim < 32 ? -1.0f : 1.0f;
        float cos_value = cos[position * 64 + dim];
        float sin_value = sin[position * 64 + dim];
        out_q[index] = q[index] * cos_value
            + sign * q[other_index] * sin_value;
        out_k[index] = k[index] * cos_value
            + sign * k[other_index] * sin_value;
    }

    int rows = in0->num / 64;
    int token_count = in0->shape[in0->shape.size() - 2];
    int patch_count = in2->shape[0];
    jt_dinov3_rope_fp32<<<rows, 64>>>(
        (const float*)in0_p, (const float*)in1_p,
        (const float*)in2_p, (const float*)in3_p,
        (float*)out0_p, (float*)out1_p,
        token_count, patch_count);
    CHECK(0 == cudaGetLastError());
    """
    return jt.code(
        [q.shape, k.shape], [q.dtype, k.dtype], [q, k, cos, sin],
        cuda_src=cuda_src,
    )


def _patch_dinov3_module(mod) -> bool:
    original = getattr(mod, "apply_rotary_pos_emb", None)
    if original is None:
        return False
    if getattr(original, "_jittor_torch_fast_dinov3_rope", False):
        mod._jittor_torch_fast_dinov3_rope = True
        return True

    def apply_rotary_pos_emb(q, k, cos, sin, **kwargs):
        if not kwargs:
            fast = _dinov3_rotary_fast_path(q, k, cos, sin)
            if fast is not None:
                return fast
        return original(q, k, cos, sin, **kwargs)

    apply_rotary_pos_emb._jittor_torch_fast_dinov3_rope = True
    apply_rotary_pos_emb._jittor_torch_original = original
    mod.apply_rotary_pos_emb = apply_rotary_pos_emb
    mod._jittor_torch_fast_dinov3_rope = True
    return True


def _patch_loaded_dense_attention() -> bool:
    mod = sys.modules.get(_DENSE_ATTENTION_MODULE)
    return mod is not None and _patch_dense_attention_module(mod)


def _patch_loaded_attention() -> bool:
    mod = sys.modules.get(_ATTENTION_MODULE)
    return mod is not None and _patch_attention_module(mod)


def _patch_loaded_sparse_attention() -> bool:
    mod = sys.modules.get(_SPARSE_ATTENTION_MODULE)
    return mod is not None and _patch_sparse_attention_module(mod)


def _patch_loaded_dinov3() -> bool:
    mod = sys.modules.get(_DINOV3_MODULE)
    return mod is not None and _patch_dinov3_module(mod)


def install() -> None:
    if _is_falsey(os.environ.get("JITTOR_TRELLIS_RUNTIME_PATCHES")):
        return
    dense_done = _patch_loaded_dense_attention()
    attention_done = _patch_loaded_attention()
    sparse_done = _patch_loaded_sparse_attention()
    dinov3_done = _patch_loaded_dinov3()
    if dense_done and attention_done and sparse_done and dinov3_done:
        return
    for finder in sys.meta_path:
        if isinstance(finder, _TrellisRuntimeFinder):
            return
    sys.meta_path.insert(0, _TrellisRuntimeFinder())


class _TrellisRuntimeLoader(importlib.abc.Loader):
    def __init__(self, loader):
        self.loader = loader

    def create_module(self, spec):
        create = getattr(self.loader, "create_module", None)
        if create is None:
            return None
        return create(spec)

    def exec_module(self, module) -> None:
        self.loader.exec_module(module)
        if module.__name__ == _DENSE_ATTENTION_MODULE:
            _patch_dense_attention_module(module)
        if module.__name__ == _ATTENTION_MODULE:
            _patch_attention_module(module)
        if module.__name__ == _SPARSE_ATTENTION_MODULE:
            _patch_sparse_attention_module(module)
        if module.__name__ == _DINOV3_MODULE:
            _patch_dinov3_module(module)


class _TrellisRuntimeFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in (
                _DENSE_ATTENTION_MODULE, _ATTENTION_MODULE,
                _SPARSE_ATTENTION_MODULE, _DINOV3_MODULE):
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return None
        if not isinstance(spec.loader, _TrellisRuntimeLoader):
            spec.loader = _TrellisRuntimeLoader(spec.loader)
        return spec
