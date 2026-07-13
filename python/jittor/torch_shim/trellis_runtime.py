"""Runtime patches for running TRELLIS.2 through the Jittor torch shim.

These patches live in Jittor rather than in TRELLIS.2 so the upstream project
and its native extension dependencies can stay unmodified.

``JITTOR_TRELLIS_CROSS_KV_CACHE=1`` opts into a sampler-scoped inference cache.
It requires eval/no-grad execution and immutable condition tensors and weights
for the duration of ``FlowEulerSampler.sample``. Concurrent samplers must not
share the same model instance while this opt-in cache is active.
``JITTOR_TRELLIS_PROCESSED_KV_CACHE=0`` keeps the projection cache active while
disabling reuse of the reshaped and normalized cross-attention keys.

``JITTOR_TRELLIS_C2S_TOPOLOGY_CACHE=1`` opts into reusing topology between
adjacent paired ``SparseChannel2Spatial`` calls. The flexible-grid mesh
finalizer is enabled for the exact TRELLIS.2 inference signature and can be
disabled with ``JITTOR_TRELLIS_FUSED_MESH=0``.

The inference RMSNorm kernels are enabled by default. Set
``JITTOR_TRELLIS_FUSED_RMS_NORM=0`` to disable both dense and sparse kernels,
or ``JITTOR_TRELLIS_FUSED_SPARSE_RMS_NORM=0`` to disable only the sparse path.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import math
import os
import sys
from contextvars import ContextVar
from typing import Dict, Tuple


_FALSEY = {"0", "false", "no", "off"}
_TRUTHY = {"1", "true", "yes", "on"}
_CU_SEQLENS_CACHE: Dict[Tuple[str, Tuple[int, ...], str], object] = {}
_DENSE_ATTENTION_MODULE = "trellis2.modules.attention.full_attn"
_ATTENTION_MODULE = "trellis2.modules.attention.modules"
_SPARSE_ATTENTION_MODULE = "trellis2.modules.sparse.attention.full_attn"
_SPARSE_ATTENTION_API_MODULE = "trellis2.modules.sparse.attention.modules"
_FLOW_EULER_MODULE = "trellis2.pipelines.samplers.flow_euler"
_C2S_MODULE = "trellis2.modules.sparse.spatial.spatial2channel"
_C2S_BLOCK_MODULE = "trellis2.models.sc_vaes.sparse_unet_vae"
_FLEXIBLE_GRID_MODULE = "o_voxel.convert.flexible_dual_grid"
_NORM_MODULE = "trellis2.modules.norm"
_DINOV3_MODULE = "transformers.models.dinov3_vit.modeling_dinov3_vit"
_LAYOUT_INFO_KEY = "_jittor_torch_layout_info"
_CU_SEQLENS_KEY = "_jittor_torch_cu_seqlens_int32"
_CROSS_KV_CONTEXT_BYTES = 1 * 1029 * 1024 * 2
_CROSS_KV_ENTRY_BYTES = 1 * 1029 * 3072 * 2
_CROSS_KV_CACHE_SCOPE = ContextVar("jittor_trellis_cross_kv_cache", default=None)
_C2S_PAIR_SCOPE = ContextVar("jittor_trellis_c2s_pair_cache", default=None)
_FLEXIBLE_GRID_OFFSETS = {}


class _CrossKVCacheState:
    def __init__(self, model, contexts, allowed, attention_allowed=None):
        self.model = model
        self.contexts = tuple(contexts)
        self.allowed = dict(allowed)
        self.attention_allowed = dict(attention_allowed or {})
        self.kv_cache = {}
        self.processed_kv_cache = {}

    def clear(self):
        self.allowed.clear()
        self.attention_allowed.clear()
        self.kv_cache.clear()
        self.processed_kv_cache.clear()


class _C2SPairCacheState:
    def __init__(self, layer):
        self.layer = layer
        self.entry = None

    def clear(self):
        self.layer = None
        self.entry = None


def _is_falsey(value) -> bool:
    return str(value or "").strip().lower() in _FALSEY


def _is_truthy(value) -> bool:
    return str(value or "").strip().lower() in _TRUTHY


def _cross_kv_cache_budget() -> int:
    try:
        value = float(os.environ.get("JITTOR_TRELLIS_CROSS_KV_CACHE_MB", "384"))
    except (TypeError, ValueError):
        value = 384.0
    if not math.isfinite(value) or value <= 0:
        return 0
    return int(min(value, 4096.0) * 1024 * 1024)


def _module_is_eval(module) -> bool:
    try:
        return not bool(module.training)
    except Exception:
        return False


def _tensor_signature(value):
    try:
        return (
            tuple(int(size) for size in value.shape),
            str(value.dtype),
            int(value.get_device()),
        )
    except Exception:
        return None


def _is_inference_source(jt, value) -> bool:
    if not isinstance(value, jt.Var):
        return False
    signature = _tensor_signature(value)
    if signature is None or signature[:2] != ((1, 1029, 1024), "float32"):
        return False
    try:
        return signature[2] >= 0 and bool(value.is_stop_grad())
    except Exception:
        return False


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


def _same_c2s_signature(left, right) -> bool:
    return (
        left[0] is right[0]
        and left[1] is right[1]
        and left[2] is right[2]
        and left[3] == right[3]
    )


def _trellis_c2s_pair_fast_path(mod, layer, x, subdivision):
    if not _is_truthy(os.environ.get("JITTOR_TRELLIS_C2S_TOPOLOGY_CACHE")):
        return None
    state = _C2S_PAIR_SCOPE.get()
    if state is None or state.layer is not layer:
        return None
    try:
        import jittor as jt

        sparse_tensor = mod.SparseTensor
        torch = mod.torch
        factor = int(layer.factor)
        coords_signature = _tensor_signature(x.coords)
        feats_signature = _tensor_signature(x.feats)
        sub_signature = _tensor_signature(subdivision.feats)
        scale = tuple(x._scale)
    except Exception:
        return None

    if not (
            isinstance(x, sparse_tensor)
            and isinstance(subdivision, sparse_tensor)
            and factor == 2
            and _module_is_eval(layer)
            and jt.flags.use_cuda
            and getattr(jt.flags, "no_grad", 0)
            and not getattr(jt.compiler, "has_acl", 0)
            and coords_signature is not None
            and coords_signature[:2] == (
                (int(x.coords.shape[0]), 4), "int32")
            and feats_signature is not None
            and len(feats_signature[0]) == 2
            and feats_signature[0][0] == coords_signature[0][0]
            and feats_signature[0][1] > 0
            and feats_signature[0][1] % 8 == 0
            and feats_signature[1] == "float16"
            and sub_signature == (
                (coords_signature[0][0], 8), "bool", coords_signature[2])
            and feats_signature[2] == coords_signature[2]
            and coords_signature[2] >= 0):
        return None
    try:
        if x.get_spatial_cache("channel2spatial_2") is not None:
            return None
    except Exception:
        return None

    signature = (layer, x.coords, subdivision.feats, scale)
    entry = state.entry
    state.entry = None
    try:
        if entry is not None and _same_c2s_signature(entry[0], signature):
            new_coords, idx, subidx = entry[1]
        elif entry is not None:
            return None
        else:
            sub = subdivision.feats
            n_leaf = sub.sum(dim=-1)
            subidx = sub.nonzero()[:, -1]
            new_coords = x.coords.clone().detach()
            new_coords[:, 1:] *= factor
            new_coords = torch.repeat_interleave(
                new_coords, n_leaf, dim=0, output_size=subidx.shape[0])
            for index in range(3):
                new_coords[:, index + 1] += (
                    subidx // factor ** index % factor)
            idx = torch.repeat_interleave(
                torch.arange(x.coords.shape[0], device=x.device),
                n_leaf, dim=0, output_size=subidx.shape[0])
            state.entry = (signature, (new_coords, idx, subidx))

        x_feats = x.feats.reshape(x.feats.shape[0] * 8, -1)
        new_feats = x_feats[idx * 8 + subidx]
        out = sparse_tensor(
            new_feats,
            new_coords,
            None if x._shape is None
            else torch.Size([x._shape[0], x._shape[1] // 8]),
        )
        out._scale = tuple(value / 2 for value in scale)
        return out
    except Exception:
        state.entry = None
        return None


def _patch_c2s_module(mod) -> bool:
    cls = getattr(mod, "SparseChannel2Spatial", None)
    original = getattr(cls, "forward", None) if cls is not None else None
    if original is None or not hasattr(mod, "SparseTensor"):
        return False
    if getattr(original, "_jittor_torch_fast_c2s_pair", False):
        mod._jittor_torch_fast_c2s_pair = True
        return True

    def forward(self, x, subdivision=None):
        if subdivision is not None:
            fast = _trellis_c2s_pair_fast_path(
                mod, self, x, subdivision)
            if fast is not None:
                return fast
        return original(self, x, subdivision)

    forward._jittor_torch_fast_c2s_pair = True
    forward._jittor_torch_original = original
    cls.forward = forward
    mod._jittor_torch_fast_c2s_pair = True
    return True


def _patch_c2s_block_module(mod) -> bool:
    cls = getattr(mod, "SparseResBlockC2S3d", None)
    original = getattr(cls, "_forward", None) if cls is not None else None
    if original is None:
        return False
    if getattr(original, "_jittor_torch_c2s_pair_scope", False):
        mod._jittor_torch_c2s_pair_scope = True
        return True

    def _forward(self, *args, **kwargs):
        if not (_is_truthy(os.environ.get(
                "JITTOR_TRELLIS_C2S_TOPOLOGY_CACHE"))
                and _module_is_eval(self)):
            return original(self, *args, **kwargs)
        try:
            import jittor as jt

            if (not jt.flags.use_cuda or not getattr(jt.flags, "no_grad", 0)
                    or getattr(jt.compiler, "has_acl", 0)):
                return original(self, *args, **kwargs)
        except Exception:
            return original(self, *args, **kwargs)

        state = _C2SPairCacheState(getattr(self, "updown", None))
        token = _C2S_PAIR_SCOPE.set(state)
        try:
            return original(self, *args, **kwargs)
        finally:
            state.clear()
            _C2S_PAIR_SCOPE.reset(token)

    _forward._jittor_torch_c2s_pair_scope = True
    _forward._jittor_torch_original = original
    cls._forward = _forward
    mod._jittor_torch_c2s_pair_scope = True
    return True


def _trellis_finalize_flexible_mesh(
        coords, dual_vertices, quad_indices, valid_rows,
        split_weight, voxel_size, aabb):
    import jittor as jt

    vertex_count = int(coords.shape[0])
    valid_count = int(valid_rows.shape[0])
    cuda_header = r"""
    template <typename C, typename V, typename S, typename A, typename O>
    __global__ void jt_trellis_flexible_vertices(
            const C* coords, const V* vertices,
            const S* voxel_size, const A* aabb, O* out,
            int64_t total) {
        int64_t index = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
        int64_t stride = (int64_t)blockDim.x * gridDim.x;
        for (; index < total; index += stride) {
            int column = (int)(index % 3);
            out[index] = (O)(((float)coords[index] + (float)vertices[index])
                * (float)voxel_size[column] + (float)aabb[column]);
        }
    }

    template <typename Q, typename R, typename W, typename O>
    __global__ void jt_trellis_flexible_faces(
            const Q* quads, const R* rows, const W* weights, O* out,
            int64_t total) {
        int64_t index = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
        int64_t stride = (int64_t)blockDim.x * gridDim.x;
        for (; index < total; index += stride) {
            int64_t row = index / 6;
            int lane = (int)(index - row * 6);
            int64_t base = (int64_t)rows[row] * 4;
            int q0 = (int)quads[base];
            int q1 = (int)quads[base + 1];
            int q2 = (int)quads[base + 2];
            int q3 = (int)quads[base + 3];
            float weight02 = (float)weights[q0] * (float)weights[q2];
            float weight13 = (float)weights[q1] * (float)weights[q3];
            int split1[6] = {0, 1, 2, 0, 2, 3};
            int split2[6] = {0, 1, 3, 3, 1, 2};
            int corner = weight02 > weight13 ? split1[lane] : split2[lane];
            out[index] = (O)quads[base + corner];
        }
    }
    """
    cuda_src = r"""
    @alias(coords, in0)
    @alias(vertices, in1)
    @alias(quads, in2)
    @alias(rows, in3)
    @alias(weights, in4)
    @alias(voxel_size, in5)
    @alias(aabb, in6)
    @alias(out_vertices, out0)
    @alias(out_faces, out1)
    int threads = 256;
    int64_t vertex_total = out_vertices->num;
    if (vertex_total) {
        int blocks = (int)((vertex_total + threads - 1) / threads);
        if (blocks > 4096) blocks = 4096;
        jt_trellis_flexible_vertices<
            coords_type, vertices_type, voxel_size_type,
            aabb_type, out_vertices_type><<<blocks, threads>>>(
            coords_p, vertices_p, voxel_size_p, aabb_p,
            out_vertices_p, vertex_total);
        CHECK(0 == cudaGetLastError());
    }
    int64_t face_total = out_faces->num;
    if (face_total) {
        int blocks = (int)((face_total + threads - 1) / threads);
        if (blocks > 4096) blocks = 4096;
        jt_trellis_flexible_faces<
            quads_type, rows_type, weights_type,
            out_faces_type><<<blocks, threads>>>(
            quads_p, rows_p, weights_p, out_faces_p, face_total);
        CHECK(0 == cudaGetLastError());
    }
    """
    return jt.code(
        [[vertex_count, 3], [valid_count * 2, 3]],
        [dual_vertices.dtype, quad_indices.dtype],
        [
            coords, dual_vertices, quad_indices, valid_rows,
            split_weight, voxel_size, aabb,
        ],
        cuda_header=cuda_header,
        cuda_src=cuda_src,
    )


def _trellis_flexible_mesh_fast_path(
        mod, coords, dual_vertices, intersected_flag, split_weight,
        aabb, voxel_size=None, grid_size=None, train=False):
    if _is_falsey(os.environ.get("JITTOR_TRELLIS_FUSED_MESH")):
        return None
    if (train is not False or voxel_size is not None
            or type(grid_size) is not int or grid_size != 512
            or type(aabb) not in (list, tuple)):
        return None
    try:
        normalized_aabb = tuple(
            tuple(float(value) for value in row) for row in aabb)
    except Exception:
        return None
    if normalized_aabb != (
            (-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)):
        return None

    try:
        import jittor as jt

        torch = mod.torch
        native = mod._C
        tensors = (coords, dual_vertices, intersected_flag, split_weight)
        if not all(isinstance(value, jt.Var) for value in tensors):
            return None
        signatures = tuple(_tensor_signature(value) for value in tensors)
        vertex_count = int(coords.shape[0])
    except Exception:
        return None
    if not (
            jt.flags.use_cuda
            and getattr(jt.flags, "no_grad", 0)
            and not getattr(jt.compiler, "has_acl", 0)
            and all(signature is not None for signature in signatures)
            and 0 < vertex_count < (1 << 30)
            and signatures[0][:2] == ((vertex_count, 3), "int32")
            and signatures[1][:2] == ((vertex_count, 3), "float32")
            and signatures[2][:2] == ((vertex_count, 3), "bool")
            and signatures[3][:2] == ((vertex_count, 1), "float32")
            and signatures[0][2] >= 0
            and len({signature[2] for signature in signatures}) == 1):
        return None

    device = coords.device
    aabb_tensor = torch.tensor(
        normalized_aabb, dtype=torch.float32, device=device)
    grid_values = (grid_size, grid_size, grid_size)
    grid_tensor = torch.tensor(
        grid_values, dtype=torch.int32, device=device)
    voxel_size = (aabb_tensor[1] - aabb_tensor[0]) / grid_tensor

    hashmap = (
        torch.full(
            (2 * vertex_count,), torch.iinfo(torch.uint32).max,
            dtype=torch.uint32, device=device),
        torch.empty(
            (2 * vertex_count,), dtype=torch.uint32, device=device),
    )
    native.hashmap_insert_3d_idx_as_val_cuda(
        *hashmap,
        torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=-1),
        *grid_values,
    )

    offset_key = signatures[0][2]
    edge_offset = _FLEXIBLE_GRID_OFFSETS.get(offset_key)
    if edge_offset is None:
        edge_offset = torch.tensor([
            [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],
            [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],
            [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]],
        ], dtype=torch.int, device=device).unsqueeze(0)
        _FLEXIBLE_GRID_OFFSETS[offset_key] = edge_offset

    edge_neighbor_voxel = (
        coords.reshape(vertex_count, 1, 1, 3) + edge_offset)
    connected_voxel = edge_neighbor_voxel[intersected_flag]
    connected_count = int(connected_voxel.shape[0])
    if connected_count:
        connected_key = torch.cat([
            torch.zeros(
                (connected_count * 4, 1), dtype=torch.int, device=device),
            connected_voxel.reshape(-1, 3),
        ], dim=1)
        connected_indices = native.hashmap_lookup_3d_cuda(
            *hashmap, connected_key, *grid_values,
        ).reshape(connected_count, 4).int()
        valid_rows = (
            (connected_indices != 0xffffffff).all(dim=1).nonzero()[:, 0])
    else:
        connected_indices = torch.empty(
            (0, 4), dtype=torch.int32, device=device)
        valid_rows = torch.empty((0,), dtype=torch.int32, device=device)
    return _trellis_finalize_flexible_mesh(
        coords, dual_vertices, connected_indices, valid_rows,
        split_weight, voxel_size, aabb_tensor,
    )


def _patch_flexible_grid_module(mod) -> bool:
    original = getattr(mod, "flexible_dual_grid_to_mesh", None)
    if original is None or not hasattr(mod, "_C") or not hasattr(mod, "torch"):
        return False
    if getattr(original, "_jittor_torch_fast_flexible_mesh", False):
        mod._jittor_torch_fast_flexible_mesh = True
        return True

    def flexible_dual_grid_to_mesh(
            coords, dual_vertices, intersected_flag, split_weight, aabb,
            voxel_size=None, grid_size=None, train=False):
        fast = _trellis_flexible_mesh_fast_path(
            mod, coords, dual_vertices, intersected_flag, split_weight,
            aabb, voxel_size=voxel_size, grid_size=grid_size, train=train)
        if fast is not None:
            return fast
        return original(
            coords, dual_vertices, intersected_flag, split_weight, aabb,
            voxel_size=voxel_size, grid_size=grid_size, train=train)

    flexible_dual_grid_to_mesh._jittor_torch_fast_flexible_mesh = True
    flexible_dual_grid_to_mesh._jittor_torch_original = original
    mod.flexible_dual_grid_to_mesh = flexible_dual_grid_to_mesh
    mod._jittor_torch_fast_flexible_mesh = True

    for name in ("o_voxel.convert", "trellis2.models.sc_vaes.fdg_vae"):
        refs = sys.modules.get(name)
        if refs is not None and getattr(
                refs, "flexible_dual_grid_to_mesh", None) is original:
            refs.flexible_dual_grid_to_mesh = flexible_dual_grid_to_mesh
    return True


def _trellis_layer_norm32_fast_path(layer, x):
    if _is_falsey(os.environ.get("JITTOR_TRELLIS_FP16_LAYERNORM")):
        return None
    try:
        import jittor as jt
        from jittor import nn

        signature = _tensor_signature(x)
        normalized_shape = tuple(
            int(size) for size in layer.normalized_shape)
        eps = float(layer.eps)
        weight = layer.weight
        bias = layer.bias
    except Exception:
        return None

    if not (
            isinstance(x, jt.Var)
            and _module_is_eval(layer)
            and jt.flags.use_cuda
            and getattr(jt.flags, "no_grad", 0)
            and not getattr(jt.compiler, "has_acl", 0)
            and signature is not None
            and len(signature[0]) == 2
            and signature[0][0] > 0
            and signature[0][1] in (64, 128, 256, 512, 1024)
            and signature[1] == "float16"
            and signature[2] >= 0
            and normalized_shape == (signature[0][1],)
            and eps == 1e-6):
        return None
    try:
        if bool(jt.is_autocast_enabled()):
            return None
    except Exception:
        return None

    if isinstance(weight, jt.Var) or isinstance(bias, jt.Var):
        if not (isinstance(weight, jt.Var) and isinstance(bias, jt.Var)):
            return None
        expected = (normalized_shape, "float32", signature[2])
        if (_tensor_signature(weight) != expected
                or _tensor_signature(bias) != expected):
            return None
    else:
        try:
            if float(weight) != 1.0 or float(bias) != 0.0:
                return None
        except Exception:
            return None
    return nn._layer_norm_no_grad_cuda(
        x, normalized_shape, weight, bias, eps)


def _patch_norm_module(mod) -> bool:
    cls = getattr(mod, "LayerNorm32", None)
    original = getattr(cls, "forward", None) if cls is not None else None
    if original is None:
        return False
    if getattr(original, "_jittor_torch_fast_fp16_layer_norm32", False):
        mod._jittor_torch_fast_fp16_layer_norm32 = True
        return True

    def forward(self, x, *args, **kwargs):
        if not args and not kwargs:
            fast = _trellis_layer_norm32_fast_path(self, x)
            if fast is not None:
                return fast
        return original(self, x, *args, **kwargs)

    forward._jittor_torch_fast_fp16_layer_norm32 = True
    forward._jittor_torch_original = original
    cls.forward = forward
    mod._jittor_torch_fast_fp16_layer_norm32 = True
    return True


def _cross_kv_record_is_active(state, record, context) -> bool:
    try:
        import jittor as jt
    except Exception:
        return False

    attention = record["attention"]
    projection = record["projection"]
    weight = getattr(projection, "weight", None)
    bias = getattr(projection, "bias", None)
    if (state.allowed.get(id(projection)) is not record
            or record["weight"] is not weight
            or record["bias"] is not bias):
        state.kv_cache.pop((id(projection), id(context)), None)
        state.processed_kv_cache.pop((id(attention), id(context)), None)
        return False
    if not (jt.flags.use_cuda and getattr(jt.flags, "no_grad", 0)):
        state.clear()
        return False
    if getattr(jt.compiler, "has_acl", 0):
        state.clear()
        return False
    if not (_module_is_eval(state.model)
            and _module_is_eval(attention)
            and _module_is_eval(projection)):
        state.clear()
        return False
    return any(value is context for value in state.contexts)


def _trellis_cached_cross_kv_projection(
        original, attention, projection, context):
    state = _CROSS_KV_CACHE_SCOPE.get()
    if state is None:
        return original(context)
    record = state.allowed.get(id(projection))
    if (record is None or record["attention"] is not attention
            or not _cross_kv_record_is_active(state, record, context)):
        return original(context)

    key = (id(projection), id(context))
    cached = state.kv_cache.get(key)
    if cached is not None:
        return cached

    output = original(context)
    context_signature = _tensor_signature(context)
    if (context_signature is None
            or _tensor_signature(output) != (
                (1, 1029, 3072), "bfloat16", context_signature[2])):
        return output
    state.kv_cache[key] = output
    return output


def _cross_attention_fast_spec(attention, device):
    if _is_falsey(os.environ.get("JITTOR_TRELLIS_PROCESSED_KV_CACHE")):
        return None
    if "forward" in getattr(attention, "__dict__", {}):
        return None
    module_name = type(attention).__module__
    class_name = type(attention).__name__
    if (module_name == _ATTENTION_MODULE
            and class_name == "MultiHeadAttention"):
        kind = "dense"
        function_name = "scaled_dot_product_attention"
    elif (module_name == _SPARSE_ATTENTION_API_MODULE
            and class_name == "SparseMultiHeadAttention"):
        kind = "sparse"
        function_name = "sparse_scaled_dot_product_attention"
    else:
        return None
    module = sys.modules.get(module_name)
    attention_func = getattr(module, function_name, None)
    k_norm = getattr(attention, "k_rms_norm", None)
    gamma = getattr(k_norm, "gamma", None)
    original = getattr(attention, "forward", None)
    if (attention_func is None or original is None
            or getattr(attention, "qk_rms_norm", None) is not True
            or not _module_is_eval(k_norm)):
        return None
    try:
        import jittor as jt

        if not isinstance(gamma, jt.Var):
            return None
        if (_tensor_signature(gamma) != ((12, 128), "float32", device)
                or abs(float(k_norm.scale) - math.sqrt(128.0)) > 1e-6):
            return None
    except Exception:
        return None
    call_original = (
        getattr(original, "_jittor_torch_original", original)
        if getattr(
            original, "_jittor_torch_trellis_cross_attention_cache", False)
        else original
    )
    return {
        "kind": kind,
        "attention_func": attention_func,
        "k_norm": k_norm,
        "gamma": gamma,
        "original": call_original,
        "had_forward": "forward" in getattr(attention, "__dict__", {}),
        "old_forward": getattr(attention, "__dict__", {}).get("forward"),
    }


def _cross_attention_projection_record(attention, device):
    if (getattr(attention, "_type", None) != "cross"
            or getattr(attention, "channels", None) != 1536
            or getattr(attention, "ctx_channels", None) != 1024
            or getattr(attention, "num_heads", None) != 12
            or getattr(attention, "head_dim", None) != 128):
        return None
    projection = getattr(attention, "to_kv", None)
    if (projection is None
            or getattr(projection, "in_features", None) != 1024
            or getattr(projection, "out_features", None) != 3072):
        return None
    try:
        import jittor as jt

        weight = projection.weight
        bias = projection.bias
        if not (_module_is_eval(attention) and _module_is_eval(projection)):
            return None
        if not all(isinstance(value, jt.Var) for value in (weight, bias)):
            return None
        if (_tensor_signature(weight) != ((3072, 1024), "bfloat16", device)
                or _tensor_signature(bias) != ((3072,), "bfloat16", device)):
            return None
    except Exception:
        return None
    original = getattr(projection, "forward", None)
    if original is None:
        return None
    call_original = (
        getattr(original, "_jittor_torch_original", original)
        if getattr(original, "_jittor_torch_trellis_cross_kv_cache", False)
        else original
    )
    return {
        "attention": attention,
        "projection": projection,
        "weight": weight,
        "bias": bias,
        "projection_original": call_original,
        "projection_had_forward": (
            "forward" in getattr(projection, "__dict__", {})),
        "projection_old_forward": getattr(
            projection, "__dict__", {}).get("forward"),
        "fast": _cross_attention_fast_spec(attention, device),
    }


def _install_cross_attention_projection(record):
    attention = record["attention"]
    projection = record["projection"]
    original = record["projection_original"]
    def forward(context, *args, **kwargs):
        if args or kwargs:
            return original(context, *args, **kwargs)
        return _trellis_cached_cross_kv_projection(
            original, attention, projection, context)

    forward._jittor_torch_trellis_cross_kv_cache = True
    forward._jittor_torch_original = original
    projection.forward = forward
    return (
        projection, forward, record["projection_had_forward"],
        record["projection_old_forward"],
    )


def _restore_cross_attention_projection(installed) -> None:
    projection, forward, had_forward, old_forward = installed
    if getattr(projection, "__dict__", {}).get("forward") is not forward:
        return
    if had_forward:
        projection.forward = old_forward
    else:
        delattr(projection, "forward")


def _trellis_cached_cross_attention(record, x, context):
    state = _CROSS_KV_CACHE_SCOPE.get()
    fast = record["fast"]
    attention = record["attention"]
    projection = record["projection"]
    if (state is None or fast is None
            or state.attention_allowed.get(id(attention)) is not record
            or not _cross_kv_record_is_active(state, record, context)):
        return None
    if (getattr(attention, "k_rms_norm", None) is not fast["k_norm"]
            or getattr(fast["k_norm"], "gamma", None) is not fast["gamma"]
            or getattr(attention, "qk_rms_norm", None) is not True):
        state.processed_kv_cache.pop(
            (id(attention), id(context)), None)
        return None
    try:
        if abs(float(fast["k_norm"].scale) - math.sqrt(128.0)) > 1e-6:
            return None
    except Exception:
        return None

    context_signature = _tensor_signature(context)
    if (context_signature is None
            or context_signature[:2] != ((1, 1029, 1024), "bfloat16")
            or context_signature[2] < 0):
        return None
    device = context_signature[2]
    try:
        if fast["kind"] == "dense":
            x_signature = _tensor_signature(x)
            if not (x_signature is not None
                    and len(x_signature[0]) == 3
                    and x_signature[0][0] == 1
                    and x_signature[0][1] > 0
                    and x_signature[0][2] == 1536
                    and x_signature[1:] == ("bfloat16", device)):
                return None
            batch, length, _ = x_signature[0]
            q = attention.to_q(x).reshape(
                batch, length, 12, 128)
        else:
            feats_signature = _tensor_signature(getattr(x, "feats", None))
            if not (feats_signature is not None
                    and len(feats_signature[0]) == 2
                    and feats_signature[0][0] > 0
                    and feats_signature[0][1] == 1536
                    and feats_signature[1:] == ("bfloat16", device)):
                return None
            q = attention._linear(attention.to_q, x)
            q = attention._reshape_chs(q, (12, -1))

        key = (id(attention), id(context))
        cached = state.processed_kv_cache.get(key)
        if cached is None:
            if fast["kind"] == "dense":
                kv = attention.to_kv(context).reshape(
                    1, 1029, 2, 12, 128)
            else:
                kv = attention._linear(attention.to_kv, context)
                kv = attention._fused_pre(kv, num_fused=2)
            k, v = kv.unbind(dim=-3)
            k = attention.k_rms_norm(k)
            expected = ((1, 1029, 12, 128), "bfloat16", device)
            if (_tensor_signature(k) != expected
                    or _tensor_signature(v) != expected):
                return None
            state.kv_cache.pop((id(projection), id(context)), None)
            cached = (k, v)
            state.processed_kv_cache[key] = cached
        k, v = cached

        q = attention.q_rms_norm(q)
        h = fast["attention_func"](q, k, v)
        if fast["kind"] == "dense":
            h = h.reshape(batch, length, -1)
            return attention.to_out(h)
        h = attention._reshape_chs(h, (-1,))
        return attention._linear(attention.to_out, h)
    except Exception:
        state.kv_cache.pop((id(projection), id(context)), None)
        state.processed_kv_cache.pop((id(attention), id(context)), None)
        return None


def _install_cross_attention_forward(record):
    fast = record["fast"]
    if fast is None:
        return None
    attention = record["attention"]
    original = fast["original"]

    def forward(x, context=None, *args, **kwargs):
        if context is not None and not args and not kwargs:
            output = _trellis_cached_cross_attention(record, x, context)
            if output is not None:
                return output
        return original(x, context, *args, **kwargs)

    forward._jittor_torch_trellis_cross_attention_cache = True
    forward._jittor_torch_original = original
    attention.forward = forward
    return (
        attention, forward, fast["had_forward"], fast["old_forward"],
    )


def _restore_cross_attention_forward(installed) -> None:
    attention, forward, had_forward, old_forward = installed
    if getattr(attention, "__dict__", {}).get("forward") is not forward:
        return
    if had_forward:
        attention.forward = old_forward
    else:
        delattr(attention, "forward")


def _patch_flow_euler_module(mod) -> bool:
    cls = getattr(mod, "FlowEulerSampler", None)
    original = getattr(cls, "sample", None) if cls is not None else None
    if original is None:
        return False
    if getattr(original, "_jittor_torch_trellis_cross_kv_scope", False):
        return True

    def sample(self, model, noise, cond=None, *args, **kwargs):
        if not _is_truthy(os.environ.get("JITTOR_TRELLIS_CROSS_KV_CACHE")):
            return original(self, model, noise, cond, *args, **kwargs)
        budget = _cross_kv_cache_budget()
        try:
            import jittor as jt
        except Exception:
            return original(self, model, noise, cond, *args, **kwargs)
        sources = [cond]
        if kwargs.get("neg_cond") is not None:
            sources.append(kwargs["neg_cond"])
        sources = list({id(value): value for value in sources}.values())
        if (budget <= 0
                or not all(_is_inference_source(jt, value) for value in sources)
                or not _module_is_eval(model)
                or str(getattr(model, "dtype", "")) != "bfloat16"):
            return original(self, model, noise, cond, *args, **kwargs)
        try:
            if bool(jt.is_autocast_enabled()):
                return original(self, model, noise, cond, *args, **kwargs)
        except Exception:
            pass
        device = _tensor_signature(sources[0])[2]
        if any(_tensor_signature(value)[2] != device for value in sources):
            return original(self, model, noise, cond, *args, **kwargs)
        try:
            modules = model.modules()
        except Exception:
            modules = ()
        records = []
        for module in modules:
            record = _cross_attention_projection_record(module, device)
            if record is not None:
                records.append(record)
        required = len(sources) * (
            _CROSS_KV_CONTEXT_BYTES + len(records) * _CROSS_KV_ENTRY_BYTES)
        if not records or budget < required:
            return original(self, model, noise, cond, *args, **kwargs)

        contexts = [value.bfloat16() for value in sources]
        if any(_tensor_signature(value) != (
                (1, 1029, 1024), "bfloat16", device) for value in contexts):
            return original(self, model, noise, cond, *args, **kwargs)
        context_by_source = {
            id(source): context for source, context in zip(sources, contexts)
        }
        cached_cond = context_by_source[id(cond)]
        cached_kwargs = dict(kwargs)
        if kwargs.get("neg_cond") is not None:
            cached_kwargs["neg_cond"] = context_by_source[id(kwargs["neg_cond"])]

        allowed = {
            id(record["projection"]): record for record in records
        }
        attention_allowed = {
            id(record["attention"]): record for record in records
            if record["fast"] is not None
        }
        state = _CrossKVCacheState(
            model, contexts, allowed, attention_allowed)
        token = _CROSS_KV_CACHE_SCOPE.set(state)
        installed_projections = []
        installed_attentions = []
        try:
            for record in records:
                installed_projections.append(
                    _install_cross_attention_projection(record))
            for record in records:
                installed = _install_cross_attention_forward(record)
                if installed is not None:
                    installed_attentions.append(installed)
            return original(
                self, model, noise, cached_cond, *args, **cached_kwargs)
        finally:
            try:
                for item in reversed(installed_attentions):
                    _restore_cross_attention_forward(item)
            finally:
                try:
                    for item in reversed(installed_projections):
                        _restore_cross_attention_projection(item)
                finally:
                    state.clear()
                    _CROSS_KV_CACHE_SCOPE.reset(token)

    sample._jittor_torch_trellis_cross_kv_scope = True
    sample._jittor_torch_original = original
    cls.sample = sample
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
    try:
        if bool(jt.is_autocast_enabled()):
            return None
    except Exception:
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
    if len(x_shape) not in (3, 4) or x_shape[-2:] != (12, 128):
        return None
    if (any(size <= 0 for size in x_shape[:-2])
            or gamma_shape != (12, 128)):
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
            if (lane == 0) {
                denominator = sqrtf(total);
                if (denominator < 1.0e-12f)
                    denominator = 1.0e-12f;
            }
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
        if not args and not kwargs and _module_is_eval(self):
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


def _patch_sparse_attention_api_module(mod) -> bool:
    cls = getattr(mod, "SparseMultiHeadRMSNorm", None)
    varlen_cls = getattr(mod, "VarLenTensor", None)
    original = getattr(cls, "forward", None) if cls is not None else None
    if original is None or not isinstance(varlen_cls, type):
        return False
    if getattr(original, "_jittor_torch_fast_sparse_trellis_rms_norm", False):
        mod._jittor_torch_fast_sparse_trellis_rms_norm = True
        return True

    def forward(self, x, *args, **kwargs):
        if (not args and not kwargs and _module_is_eval(self)
                and not _is_falsey(os.environ.get(
                    "JITTOR_TRELLIS_FUSED_SPARSE_RMS_NORM"))):
            wrapped = isinstance(x, varlen_cls)
            value = getattr(x, "feats", None) if wrapped else x
            fast = _trellis_multihead_rms_norm_fast_path(
                value, getattr(self, "gamma", None),
                getattr(self, "scale", None))
            if fast is not None:
                if not wrapped:
                    return fast
                try:
                    return x.replace(fast)
                except Exception:
                    pass
        return original(self, x, *args, **kwargs)

    forward._jittor_torch_fast_sparse_trellis_rms_norm = True
    forward._jittor_torch_original = original
    cls.forward = forward
    mod._jittor_torch_fast_sparse_trellis_rms_norm = True
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


def _patch_loaded_sparse_attention_api() -> bool:
    mod = sys.modules.get(_SPARSE_ATTENTION_API_MODULE)
    return mod is not None and _patch_sparse_attention_api_module(mod)


def _patch_loaded_flow_euler() -> bool:
    mod = sys.modules.get(_FLOW_EULER_MODULE)
    return mod is not None and _patch_flow_euler_module(mod)


def _patch_loaded_c2s() -> bool:
    mod = sys.modules.get(_C2S_MODULE)
    return mod is not None and _patch_c2s_module(mod)


def _patch_loaded_c2s_block() -> bool:
    mod = sys.modules.get(_C2S_BLOCK_MODULE)
    return mod is not None and _patch_c2s_block_module(mod)


def _patch_loaded_flexible_grid() -> bool:
    mod = sys.modules.get(_FLEXIBLE_GRID_MODULE)
    return mod is not None and _patch_flexible_grid_module(mod)


def _patch_loaded_norm() -> bool:
    mod = sys.modules.get(_NORM_MODULE)
    return mod is not None and _patch_norm_module(mod)


def _patch_loaded_dinov3() -> bool:
    mod = sys.modules.get(_DINOV3_MODULE)
    return mod is not None and _patch_dinov3_module(mod)


def install() -> None:
    if _is_falsey(os.environ.get("JITTOR_TRELLIS_RUNTIME_PATCHES")):
        return
    dense_done = _patch_loaded_dense_attention()
    attention_done = _patch_loaded_attention()
    sparse_done = _patch_loaded_sparse_attention()
    sparse_api_done = _patch_loaded_sparse_attention_api()
    sampler_done = _patch_loaded_flow_euler()
    c2s_done = _patch_loaded_c2s()
    c2s_block_done = _patch_loaded_c2s_block()
    flexible_grid_done = _patch_loaded_flexible_grid()
    norm_done = _patch_loaded_norm()
    dinov3_done = _patch_loaded_dinov3()
    if (dense_done and attention_done and sparse_done and sparse_api_done
            and sampler_done and c2s_done and c2s_block_done
            and flexible_grid_done and norm_done
            and dinov3_done):
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
        if module.__name__ == _SPARSE_ATTENTION_API_MODULE:
            _patch_sparse_attention_api_module(module)
        if module.__name__ == _FLOW_EULER_MODULE:
            _patch_flow_euler_module(module)
        if module.__name__ == _C2S_MODULE:
            _patch_c2s_module(module)
        if module.__name__ == _C2S_BLOCK_MODULE:
            _patch_c2s_block_module(module)
        if module.__name__ == _FLEXIBLE_GRID_MODULE:
            _patch_flexible_grid_module(module)
        if module.__name__ == _NORM_MODULE:
            _patch_norm_module(module)
        if module.__name__ == _DINOV3_MODULE:
            _patch_dinov3_module(module)


class _TrellisRuntimeFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in (
                _DENSE_ATTENTION_MODULE, _ATTENTION_MODULE,
                _SPARSE_ATTENTION_MODULE, _SPARSE_ATTENTION_API_MODULE,
                _FLOW_EULER_MODULE,
                _C2S_MODULE, _C2S_BLOCK_MODULE, _FLEXIBLE_GRID_MODULE,
                _NORM_MODULE,
                _DINOV3_MODULE):
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return None
        if not isinstance(spec.loader, _TrellisRuntimeLoader):
            spec.loader = _TrellisRuntimeLoader(spec.loader)
        return spec
