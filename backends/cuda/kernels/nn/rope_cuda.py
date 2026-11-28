"""CUDA inference kernels for rotary position embeddings."""
from jittor._core.dtypes import dtype_name as _jittor_dtype_name

import jittor as jt
from jittor._runtime.core_api import _output_requires_grad, _stop_grad_outputs
from jittor._runtime.dispatch import optional_kernel

from ._inference import cached_source


def _rotary_embedding_contract(
    positions,
    q,
    k,
    cos_sin_cache,
    *,
    head_size,
    rotary_dim,
    is_neox_style,
):
    tensors = (positions, q, k, cos_sin_cache)
    if not all(isinstance(value, jt.Var) for value in tensors):
        return None
    if _output_requires_grad(tensors):
        return None
    if not is_neox_style:
        return None
    try:
        q_shape = tuple(int(size) for size in q.shape)
        k_shape = tuple(int(size) for size in k.shape)
        cache_shape = tuple(int(size) for size in cos_sin_cache.shape)
        token_count = int(positions.numel())
        head_size = int(head_size)
        rotary_dim = int(rotary_dim)
    except Exception:
        return None
    if len(q_shape) != 2 or len(k_shape) != 2 or len(cache_shape) != 2:
        return None
    if q_shape[0] != token_count or k_shape[0] != token_count:
        return None
    if head_size <= 0 or q_shape[1] % head_size or k_shape[1] % head_size:
        return None
    if rotary_dim <= 0 or rotary_dim > head_size or rotary_dim % 2:
        return None
    if cache_shape[0] <= 0 or cache_shape[1] != rotary_dim:
        return None
    value_dtypes = {_jittor_dtype_name(value.dtype) for value in (q, k, cos_sin_cache)}
    if len(value_dtypes) != 1 or value_dtypes.pop() not in (
        "float16", "bfloat16", "float32"
    ):
        return None
    if _jittor_dtype_name(positions.dtype) not in ("int32", "int64"):
        return None
    return q_shape, k_shape, cache_shape, head_size, rotary_dim


def _rotary_embedding_supported(*args, **kwargs):
    return _rotary_embedding_contract(*args, **kwargs) is not None


@optional_kernel("nn.rotary_embedding.inference", ("cuda", "rocm_legacy", "corex_legacy"),
                 supports=_rotary_embedding_supported)
def _rotary_embedding_cuda(
    positions, q, k, cos_sin_cache, *, head_size, rotary_dim, is_neox_style,
):
    """Inference-only CUDA RoPE for GQA query/key shapes."""
    q_shape, k_shape, cache_shape, head_size, rotary_dim = _rotary_embedding_contract(
        positions, q, k, cos_sin_cache, head_size=head_size,
        rotary_dim=rotary_dim, is_neox_style=is_neox_style)

    cuda_src = cached_source(r"""
    __global__ static void rotary_embedding(
            const in0_type* positions, const in1_type* q, const in2_type* k,
            const in3_type* cache, out0_type* out_q, out1_type* out_k,
            int64_t q_total, int64_t k_total) {
        // Keep the real and imaginary products in the same FP32 arithmetic
        // order as the standalone CUDA rotation, including its final fma.
        auto rotate = [](float value, float other, float c, float s, bool first) {
            if (%(float32)d)
                return first ? fmaf(value, c, -(other * s)) : fmaf(other, s, value * c);
            return value * c + (first ? -1.0f : 1.0f) * other * s;
        };
        int64_t index = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
        int64_t stride = (int64_t)blockDim.x * gridDim.x;
        int64_t total = q_total > k_total ? q_total : k_total;
        for (; index < total; index += stride) {
            if (index < q_total) {
                int dim = (int)(index %% %(head_size)d);
                int token = (int)(index / %(q_token_stride)d);
                if (dim >= %(rotary_dim)d) {
                    out_q[index] = q[index];
                } else {
                    int position = (int)positions[token];
                    int half = %(rotary_dim)d / 2;
                    int other_dim = dim < half ? dim + half : dim - half;
                    int64_t other_index = index - dim + other_dim;
                    int frequency_dim = dim < half ? dim : dim - half;
                    float cos_value = static_cast<float>(
                        cache[position * %(cache_width)d + frequency_dim]);
                    float sin_value = static_cast<float>(
                        cache[position * %(cache_width)d + half + frequency_dim]);
                    out_q[index] = out0_type(rotate(static_cast<float>(q[index]),
                        static_cast<float>(q[other_index]), cos_value, sin_value, dim < half));
                }
            }
            if (index < k_total) {
                int dim = (int)(index %% %(head_size)d);
                int token = (int)(index / %(k_token_stride)d);
                if (dim >= %(rotary_dim)d) {
                    out_k[index] = k[index];
                } else {
                    int position = (int)positions[token];
                    int half = %(rotary_dim)d / 2;
                    int other_dim = dim < half ? dim + half : dim - half;
                    int64_t other_index = index - dim + other_dim;
                    int frequency_dim = dim < half ? dim : dim - half;
                    float cos_value = static_cast<float>(
                        cache[position * %(cache_width)d + frequency_dim]);
                    float sin_value = static_cast<float>(
                        cache[position * %(cache_width)d + half + frequency_dim]);
                    out_k[index] = out1_type(rotate(static_cast<float>(k[index]),
                        static_cast<float>(k[other_index]), cos_value, sin_value, dim < half));
                }
            }
        }
    }
    int64_t q_total = in1->num;
    int64_t k_total = in2->num;
    int64_t total = q_total > k_total ? q_total : k_total;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    if (blocks > 4096) blocks = 4096;
    if (total) rotary_embedding<<<blocks, threads>>>(
        in0_p, in1_p, in2_p, in3_p, out0_p, out1_p, q_total, k_total);
    CHECK(0 == cudaGetLastError());
    """, {
        "float32": int(_jittor_dtype_name(q.dtype) == "float32"),
        "cache_width": cache_shape[1],
        "head_size": head_size,
        "k_token_stride": k_shape[1],
        "q_token_stride": q_shape[1],
        "rotary_dim": rotary_dim,
    })
    return _stop_grad_outputs(jt.code(
        [q.shape, k.shape], [q.dtype, k.dtype],
        [positions, q, k, cos_sin_cache], cuda_src=cuda_src,
    ))


def _partial_rotary_embedding_contract(q, k, cos, sin, *, prefix_tokens, rotary_dim=None):
    tensors = (q, k, cos, sin)
    if not all(isinstance(value, jt.Var) for value in tensors):
        return None
    if _output_requires_grad(tensors):
        return None
    dtypes = tuple(_jittor_dtype_name(value.dtype) for value in tensors)
    if len(set(dtypes)) != 1 or dtypes[0] != "float32":
        return None
    try:
        q_shape = tuple(int(size) for size in q.shape)
        k_shape = tuple(int(size) for size in k.shape)
        cos_shape = tuple(int(size) for size in cos.shape)
        sin_shape = tuple(int(size) for size in sin.shape)
        prefix_count = int(prefix_tokens)
    except Exception:
        return None
    if q_shape != k_shape or len(q_shape) < 2 or cos_shape != sin_shape:
        return None
    if len(cos_shape) != 2 or any(size <= 0 for size in q_shape[:-2]):
        return None

    token_count, head_dim = q_shape[-2:]
    rotated_tokens, table_dim = cos_shape
    rotate = table_dim if rotary_dim is None else int(rotary_dim)
    if (
        prefix_count < 0
        or prefix_count + rotated_tokens != token_count
        or rotate <= 0
        or rotate > head_dim
        or rotate > table_dim
        or rotate % 2
        or int(cos.numel()) != rotated_tokens * table_dim
    ):
        return None
    return head_dim, token_count, prefix_count, rotate, table_dim


def _partial_rotary_embedding_supported(*args, **kwargs):
    return _partial_rotary_embedding_contract(*args, **kwargs) is not None


@optional_kernel("nn.partial_rotary_embedding.inference", ("cuda", "rocm_legacy", "corex_legacy"),
                 supports=_partial_rotary_embedding_supported)
def partial_rotary_embedding_cuda(q, k, cos, sin, *, prefix_tokens, rotary_dim=None):
    """Rotate final q/k tokens after a prefix, preserving remaining channels.

    Token and channel axes are the final two axes; cos/sin contain the rotated
    tokens. Unsupported inference contracts return None.
    """
    head_dim, token_count, prefix_count, rotate, table_dim = _partial_rotary_embedding_contract(
        q, k, cos, sin, prefix_tokens=prefix_tokens, rotary_dim=rotary_dim)

    cuda_src = cached_source(r"""
    __global__ static void partial_rope(
            const in0_type* q, const in1_type* k,
            const in2_type* cos, const in3_type* sin,
            out0_type* out_q, out1_type* out_k, int64_t total) {
        int64_t index = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
        int64_t stride = (int64_t)blockDim.x * gridDim.x;
        for (; index < total; index += stride) {
            int dim = (int)(index %% %(head_dim)d);
            int token = (int)((index / %(head_dim)d) %% %(token_count)d);
            if (token < %(prefix_count)d || dim >= %(rotate)d) {
                out_q[index] = q[index];
                out_k[index] = k[index];
                continue;
            }
            int position = token - %(prefix_count)d;
            int half = %(rotate)d / 2;
            int other_dim = dim < half ? dim + half : dim - half;
            int64_t other_index = index - dim + other_dim;
            float sign = dim < half ? -1.0f : 1.0f;
            float cos_value = static_cast<float>(
                cos[position * %(table_dim)d + dim]);
            float sin_value = static_cast<float>(
                sin[position * %(table_dim)d + dim]);
            out_q[index] = out0_type(static_cast<float>(q[index]) * cos_value
                + sign * static_cast<float>(q[other_index]) * sin_value);
            out_k[index] = out1_type(static_cast<float>(k[index]) * cos_value
                + sign * static_cast<float>(k[other_index]) * sin_value);
        }
    }
    int64_t total = in0->num;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    if (blocks > 4096) blocks = 4096;
    if (total) partial_rope<<<blocks, threads>>>(
        in0_p, in1_p, in2_p, in3_p, out0_p, out1_p, total);
    CHECK(0 == cudaGetLastError());
    """, {
        "head_dim": head_dim,
        "token_count": token_count,
        "prefix_count": prefix_count,
        "rotate": rotate,
        "table_dim": table_dim,
    })
    return _stop_grad_outputs(jt.code(
        [q.shape, k.shape],
        [q.dtype, k.dtype],
        [q, k, cos, sin],
        cuda_src=cuda_src,
    ))


def _is_seq_major_strides(x):
    if int(x._storage_offset()) != 0:
        return False
    b, h, s, d = (int(v) for v in x.shape)
    st = tuple(int(v) for v in x._storage_strides())
    return (
        len(st) == 4
        and st[3] == 1
        and (h == 1 or st[1] == d)
        and (s == 1 or st[2] == h * d)
        and (b == 1 or st[0] == s * h * d)
    )


def _neox_rope_physical(x):
    """Return ``(dense_source, seq_major)`` for a 4D tensor or ``(0,2,1,3)`` transpose view."""
    if not isinstance(x, jt.Var) or len(x.shape) != 4:
        return None
    if x._storage_is_contiguous():
        return x, False
    axes = getattr(x, "_transpose_view_axes", None)
    if axes is not None and tuple(axes()) == (0, 2, 1, 3):
        src = x._transpose_view_source()
        if isinstance(src, jt.Var) and len(src.shape) == 4:
            return src, True
    if tuple(x._producer_transpose_axes()) == (0, 2, 1, 3):
        src = x._input(0)
        if isinstance(src, jt.Var) and len(src.shape) == 4:
            return src, True
    if _is_seq_major_strides(x):
        src = x.transpose(0, 2, 1, 3)
        if isinstance(src, jt.Var) and len(src.shape) == 4:
            return src, True
    if not x.is_finished and x._producer_name() not in ("transpose", "reindex"):
        return x, False
    return None


def _unwrap_broadcast_clone(t):
    while isinstance(t, jt.Var) and t._producer_name() in ("broadcast_to", "clone"):
        t = t._input(0)
    return t


def _neox_rope_contract(x, cos, sin):
    if not all(isinstance(v, jt.Var) for v in (x, cos, sin)):
        return None
    if _output_requires_grad((x, cos, sin)):
        return None
    phys = _neox_rope_physical(x)
    if phys is None:
        return None
    x_dense, seq_major = phys
    cos = _unwrap_broadcast_clone(cos)
    sin = _unwrap_broadcast_clone(sin)
    if not (isinstance(cos, jt.Var) and isinstance(sin, jt.Var)):
        return None
    dtype = _jittor_dtype_name(x.dtype)
    if dtype not in ("float16", "bfloat16", "float32"):
        return None
    if _jittor_dtype_name(cos.dtype) != dtype or _jittor_dtype_name(sin.dtype) != dtype:
        return None
    x_shape = tuple(int(s) for s in x.shape)
    b, h, s, d = x_shape
    if b <= 0 or h <= 0 or s <= 0 or d <= 0 or (d % 2) != 0:
        return None
    c_shape = tuple(int(v) for v in cos.shape)
    s_shape = tuple(int(v) for v in sin.shape)
    if c_shape != s_shape or len(c_shape) > 4:
        return None
    c_shape = (1,) * (4 - len(c_shape)) + c_shape
    if c_shape[3] != d:
        return None
    for ci, xi in zip(c_shape[:3], x_shape[:3]):
        if ci not in (1, xi):
            return None
    return x_dense, seq_major, cos, sin, x_shape, c_shape, dtype


def _neox_rope_supported(x, cos, sin):
    return _neox_rope_contract(x, cos, sin) is not None


@optional_kernel("nn.neox_rope.inference", "cuda", supports=_neox_rope_supported)
def _neox_rope_cuda(x, cos, sin):
    """Fused vectorized CUDA NeoX RoPE: ``x * cos + rotate_half(x) * sin``."""
    contract = _neox_rope_contract(x, cos, sin)
    if contract is None:
        return None
    x_dense, seq_major, cos, sin, (b, h, s, d), (c0, c1, c2, _), dtype = contract
    half = d // 2
    vec2 = 1 if (dtype in ("float16", "bfloat16") and half % 8 == 0) else 0
    vec4 = 1 if (dtype == "float32" and half % 4 == 0) else 0
    half_vec2 = max(1, half // 8)
    half_vec4 = max(1, half // 4)
    cuda_src = cached_source(r"""
    __global__ static void neox_rope_vec2(
            const in0_type* __restrict__ x,
            const in1_type* __restrict__ cos_p,
            const in2_type* __restrict__ sin_p,
            out0_type* __restrict__ y,
            int B, int H, int S, int c0, int c1, int c2) {
        if (sizeof(in0_type) == 2 && %(vec2)d) {
            constexpr int kHalfVec = %(half_vec2)d;
            constexpr int kFullVec = kHalfVec * 2;
            long long total_vecs = (long long)B * H * S * kHalfVec;
            long long stride = (long long)blockDim.x * gridDim.x;
            const uint4* x8 = reinterpret_cast<const uint4*>(x);
            const uint4* c8 = reinterpret_cast<const uint4*>(cos_p);
            const uint4* s8 = reinterpret_cast<const uint4*>(sin_p);
            uint4* y8 = reinterpret_cast<uint4*>(y);
            for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
                 idx < total_vecs; idx += stride) {
                int v = (int)(idx %% kHalfVec);
                long long row = idx / kHalfVec;
                int si = (int)(row %% S);
                long long bh = row / S;
                int hi = (int)(bh %% H);
                int bi = (int)(bh / H);
                long long in_row = %(seq_major)d
                    ? (((long long)bi * S + si) * H + hi)
                    : row;
                long long cs_row = ((long long)(c0 > 1 ? bi : 0) * c1 + (c1 > 1 ? hi : 0)) * c2
                    + (c2 > 1 ? si : 0);
                uint4 xr1 = x8[in_row * kFullVec + v];
                uint4 xr2 = x8[in_row * kFullVec + kHalfVec + v];
                uint4 cr1 = c8[cs_row * kFullVec + v];
                uint4 cr2 = c8[cs_row * kFullVec + kHalfVec + v];
                uint4 sr1 = s8[cs_row * kFullVec + v];
                uint4 sr2 = s8[cs_row * kFullVec + kHalfVec + v];
                const in0_type* x1p = reinterpret_cast<const in0_type*>(&xr1);
                const in0_type* x2p = reinterpret_cast<const in0_type*>(&xr2);
                const in1_type* c1p = reinterpret_cast<const in1_type*>(&cr1);
                const in1_type* c2p = reinterpret_cast<const in1_type*>(&cr2);
                const in2_type* s1p = reinterpret_cast<const in2_type*>(&sr1);
                const in2_type* s2p = reinterpret_cast<const in2_type*>(&sr2);
                uint4 yr1, yr2;
                out0_type* y1p = reinterpret_cast<out0_type*>(&yr1);
                out0_type* y2p = reinterpret_cast<out0_type*>(&yr2);
                #pragma unroll
                for (int k = 0; k < 8; ++k) {
                    float a1 = static_cast<float>(x1p[k]);
                    float a2 = static_cast<float>(x2p[k]);
                    out0_type p1a = out0_type(a1 * static_cast<float>(c1p[k]));
                    out0_type p1b = out0_type((-a2) * static_cast<float>(s1p[k]));
                    y1p[k] = out0_type(static_cast<float>(p1a) + static_cast<float>(p1b));
                    out0_type p2a = out0_type(a2 * static_cast<float>(c2p[k]));
                    out0_type p2b = out0_type(a1 * static_cast<float>(s2p[k]));
                    y2p[k] = out0_type(static_cast<float>(p2a) + static_cast<float>(p2b));
                }
                y8[row * kFullVec + v] = yr1;
                y8[row * kFullVec + kHalfVec + v] = yr2;
            }
        }
    }
    __global__ static void neox_rope_vec4(
            const in0_type* __restrict__ x,
            const in1_type* __restrict__ cos_p,
            const in2_type* __restrict__ sin_p,
            out0_type* __restrict__ y,
            int B, int H, int S, int c0, int c1, int c2) {
        if (sizeof(in0_type) == 4 && %(vec4)d) {
            constexpr int kHalfVec = %(half_vec4)d;
            constexpr int kFullVec = kHalfVec * 2;
            long long total_vecs = (long long)B * H * S * kHalfVec;
            long long stride = (long long)blockDim.x * gridDim.x;
            const float4* x4 = reinterpret_cast<const float4*>(x);
            const float4* c4 = reinterpret_cast<const float4*>(cos_p);
            const float4* s4 = reinterpret_cast<const float4*>(sin_p);
            float4* y4 = reinterpret_cast<float4*>(y);
            for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
                 idx < total_vecs; idx += stride) {
                int v = (int)(idx %% kHalfVec);
                long long row = idx / kHalfVec;
                int si = (int)(row %% S);
                long long bh = row / S;
                int hi = (int)(bh %% H);
                int bi = (int)(bh / H);
                long long in_row = %(seq_major)d
                    ? (((long long)bi * S + si) * H + hi)
                    : row;
                long long cs_row = ((long long)(c0 > 1 ? bi : 0) * c1 + (c1 > 1 ? hi : 0)) * c2
                    + (c2 > 1 ? si : 0);
                float4 x1 = x4[in_row * kFullVec + v];
                float4 x2 = x4[in_row * kFullVec + kHalfVec + v];
                float4 c1v = c4[cs_row * kFullVec + v];
                float4 c2v = c4[cs_row * kFullVec + kHalfVec + v];
                float4 s1v = s4[cs_row * kFullVec + v];
                float4 s2v = s4[cs_row * kFullVec + kHalfVec + v];
                y4[row * kFullVec + v] = make_float4(
                    x1.x * c1v.x + (-x2.x) * s1v.x,
                    x1.y * c1v.y + (-x2.y) * s1v.y,
                    x1.z * c1v.z + (-x2.z) * s1v.z,
                    x1.w * c1v.w + (-x2.w) * s1v.w);
                y4[row * kFullVec + kHalfVec + v] = make_float4(
                    x2.x * c2v.x + x1.x * s2v.x,
                    x2.y * c2v.y + x1.y * s2v.y,
                    x2.z * c2v.z + x1.z * s2v.z,
                    x2.w * c2v.w + x1.w * s2v.w);
            }
        }
    }
    __global__ static void neox_rope_scalar(
            const in0_type* __restrict__ x,
            const in1_type* __restrict__ cos_p,
            const in2_type* __restrict__ sin_p,
            out0_type* __restrict__ y,
            int B, int H, int S, int c0, int c1, int c2) {
        constexpr int kHalf = %(half)d;
        constexpr int kD = %(d)d;
        long long total = (long long)B * H * S * kHalf;
        long long stride = (long long)blockDim.x * gridDim.x;
        for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
             idx < total; idx += stride) {
            int j = (int)(idx %% kHalf);
            long long row = idx / kHalf;
            int si = (int)(row %% S);
            long long bh = row / S;
            int hi = (int)(bh %% H);
            int bi = (int)(bh / H);
            long long in_row = %(seq_major)d
                ? (((long long)bi * S + si) * H + hi)
                : row;
            long long cs_row = ((long long)(c0 > 1 ? bi : 0) * c1 + (c1 > 1 ? hi : 0)) * c2
                + (c2 > 1 ? si : 0);
            float a1 = static_cast<float>(x[in_row * kD + j]);
            float a2 = static_cast<float>(x[in_row * kD + kHalf + j]);
            float c1v = static_cast<float>(cos_p[cs_row * kD + j]);
            float c2v = static_cast<float>(cos_p[cs_row * kD + kHalf + j]);
            float s1v = static_cast<float>(sin_p[cs_row * kD + j]);
            float s2v = static_cast<float>(sin_p[cs_row * kD + kHalf + j]);
            if (sizeof(out0_type) == 2) {
                out0_type p1a = out0_type(a1 * c1v);
                out0_type p1b = out0_type((-a2) * s1v);
                y[row * kD + j] = out0_type(static_cast<float>(p1a) + static_cast<float>(p1b));
                out0_type p2a = out0_type(a2 * c2v);
                out0_type p2b = out0_type(a1 * s2v);
                y[row * kD + kHalf + j] = out0_type(static_cast<float>(p2a) + static_cast<float>(p2b));
            } else {
                y[row * kD + j] = out0_type(a1 * c1v + (-a2) * s1v);
                y[row * kD + kHalf + j] = out0_type(a2 * c2v + a1 * s2v);
            }
        }
    }
    int B = (int)out0->shape[0];
    int H = (int)out0->shape[1];
    int S = (int)out0->shape[2];
    int c0 = %(c0_gt1)d ? B : 1;
    int c1 = %(c1_gt1)d ? H : 1;
    int c2 = %(c2_gt1)d ? S : 1;
    bool aligned16 = ((((size_t)in0_p | (size_t)in1_p | (size_t)in2_p | (size_t)out0_p) & 15u) == 0);
    if (sizeof(in0_type) == 2 && %(vec2)d && aligned16) {
        long long tv = (long long)B * H * S * %(half_vec2)d;
        int blocks = (int)((tv + 255) / 256);
        if (blocks > 65535) blocks = 65535;
        if (tv > 0) neox_rope_vec2<<<blocks, 256>>>(in0_p, in1_p, in2_p, out0_p, B, H, S, c0, c1, c2);
    } else if (sizeof(in0_type) == 4 && %(vec4)d && aligned16) {
        long long tv = (long long)B * H * S * %(half_vec4)d;
        int blocks = (int)((tv + 255) / 256);
        if (blocks > 65535) blocks = 65535;
        if (tv > 0) neox_rope_vec4<<<blocks, 256>>>(in0_p, in1_p, in2_p, out0_p, B, H, S, c0, c1, c2);
    } else {
        long long total = (long long)B * H * S * %(half)d;
        int blocks = (int)((total + 255) / 256);
        if (blocks > 65535) blocks = 65535;
        if (total > 0) neox_rope_scalar<<<blocks, 256>>>(in0_p, in1_p, in2_p, out0_p, B, H, S, c0, c1, c2);
    }
    CHECK(0 == cudaGetLastError());
    """, {
        "d": d,
        "half": half,
        "vec2": vec2,
        "vec4": vec4,
        "half_vec2": half_vec2,
        "half_vec4": half_vec4,
        "seq_major": int(seq_major),
        "c0_gt1": int(c0 > 1),
        "c1_gt1": int(c1 > 1),
        "c2_gt1": int(c2 > 1),
    })
    return _stop_grad_outputs(jt.code(
        x.shape, x.dtype, [x_dense, cos, sin], cuda_src=cuda_src,
    ))


__all__ = ["partial_rotary_embedding_cuda"]
