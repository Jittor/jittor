"""CUDA inference kernels for rotary position embeddings."""
from functools import lru_cache

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


def _neox_rope_supported(x, cos, sin):
    return jt.core._neox_rope(x, cos, sin) is not None


@optional_kernel("nn.neox_rope.inference", "cuda", supports=_neox_rope_supported)
def _neox_rope_cuda(x, cos, sin):
    """Fused vectorized CUDA NeoX RoPE: ``x * cos + rotate_half(x) * sin``."""
    out = jt.core._neox_rope(x, cos, sin, _neox_rope_source)
    if out is not None and jt.flags.keep_graph == 2 and _jittor_dtype_name(x.dtype) == "float32":
        # The CodeOp inputs include the original broadcast factors. Keep the
        # logical rotation input as well, so concatenation can read its view
        # directly if the rotation has not executed.
        _, cosine, sine = out._inputs()
        operands = x, cosine, sine
        versions = tuple(value._saved_storage_version() for value in operands)
        out._jt_pending_neox_rope = (out.var_ptr, operands, versions)
    return out


@optional_kernel("nn.neox_rope_concat.inference", "cuda", dtypes=("float32",))
def _neox_rope_concat(prefix, rotated):
    tag = getattr(rotated, "_jt_pending_neox_rope", None)
    if tag is None or rotated.is_finished or tag[0] != rotated.var_ptr:
        return None
    _, rope_operands, versions = tag
    x, cosine, sine = rope_operands
    if (prefix.ndim != 4 or prefix.dtype != rotated.dtype
            or tuple(t._saved_storage_version() for t in rope_operands) != versions
            or prefix.shape[:2] != x.shape[:2] or prefix.shape[3] != x.shape[3]
            or _output_requires_grad(prefix, x, cosine, sine)):
        return None
    prefix_count, x_count, width = (int(prefix.shape[2]), int(x.shape[2]), int(x.shape[3]))
    if not prefix_count or any(size <= 0 for size in prefix.shape):
        return None
    layouts = [_interleaved_rope_storage(t) for t in (prefix, x)]
    if any(layout is None for layout in layouts):
        return None
    operands, strides, offsets = zip(*layouts)
    prefix_strides, x_strides = (s or tuple(t._storage_strides())
                                 for s, t in zip(strides, (prefix, x)))
    factor_shape = (1,) * (4 - cosine.ndim) + tuple(cosine.shape)
    result_shape = (*x.shape[:2], prefix_count + x_count, width)
    header, launch = _neox_concat_source(result_shape, prefix_count, prefix_strides, x_strides,
                                         offsets, factor_shape)
    return jt.code(result_shape, x.dtype, [*operands, cosine, sine],
                   cuda_header=header, cuda_src=launch, backend="cuda")


@lru_cache(maxsize=128)
def _neox_concat_source(shape, prefix_count, prefix_strides, x_strides, offsets, factors):
    _, heads, length, width = shape
    half = width // 2
    header = f"""
    __global__ static void neox_rope_concat(const float* prefix, const float* x,
            const float* cos, const float* sin, float* out, int64_t count) {{
        const int64_t stride = int64_t(blockDim.x) * gridDim.x;
        for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += stride) {{
            int j = i % {half};
            int64_t row = i / {half};
            int token = row % {length};
            int head = row / {length} % {heads};
            int64_t batch = row / {length * heads};
            float* dst = out + row * {width} + j;
            if (token < {prefix_count}) {{
                const float* src = prefix + {offsets[0]} + batch * {prefix_strides[0]}LL
                    + head * {prefix_strides[1]}LL + token * {prefix_strides[2]}LL + j;
                dst[0] = src[0];
                dst[{half}] = src[{half}];
            }} else {{
                token -= {prefix_count};
                const float* src = x + {offsets[1]} + batch * {x_strides[0]}LL
                    + head * {x_strides[1]}LL + token * {x_strides[2]}LL + j;
                const int64_t factor = (batch * {int(factors[0] > 1)} * {factors[1] * factors[2]}LL
                    + head * {int(factors[1] > 1)} * {factors[2]}LL
                    + token * {int(factors[2] > 1)}) * {width} + j;
                float lo = src[0], hi = src[{half}];
                dst[0] = lo * cos[factor] + (-hi) * sin[factor];
                dst[{half}] = hi * cos[factor + {half}] + lo * sin[factor + {half}];
            }}
        }}
    }}
    """
    launch = """
    int64_t count = out0->num / 2;
    int blocks = (int)std::min<int64_t>((count + 127) / 128, 65535);
    neox_rope_concat<<<blocks, 128>>>(in0_p, in1_p, in2_p, in3_p, out0_p, count);
    CHECK(0 == cudaGetLastError());
    """
    return header, launch


def _neox_rope_source(d, float32, seq_major, c0_gt1, c1_gt1, c2_gt1):
    half = d // 2
    vec2 = int(not float32 and half % 8 == 0)
    vec4 = int(float32 and half % 4 == 0)
    half_vec2 = max(1, half // 8)
    half_vec4 = max(1, half // 4)
    return r"""
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
    """ % {
        "d": d,
        "half": half,
        "vec2": vec2,
        "vec4": vec4,
        "half_vec2": half_vec2,
        "half_vec4": half_vec4,
        "seq_major": int(seq_major),
        "c0_gt1": c0_gt1,
        "c1_gt1": c1_gt1,
        "c2_gt1": c2_gt1,
    }


def _interleaved_rope_storage(x):
    if x._storage_is_contiguous():
        return x, (), 0
    strides = tuple(x._storage_strides())
    if strides[-1] != 1 or any(s <= 0 for s in strides):
        return None
    base = x
    while not base._storage_is_contiguous() and not base.is_finished:
        if not base._producer_is_view():
            return None
        inputs = base._inputs()
        if len(inputs) != 1:
            return None
        base = inputs[0]
    offset = x._storage_offset() - base._storage_offset()
    if not base._storage_is_contiguous():
        if not jt.flags.no_grad or base.location() != "device":
            return None
        base = base._physical_storage_window()
    extent = sum((dim - 1) * stride for dim, stride in zip(x.shape, strides))
    if base.dtype != x.dtype or offset < 0 or offset + extent >= base.numel():
        return None
    return base, strides, offset


def _interleaved_rope_contract(x, cos, sin):
    if not all(isinstance(value, jt.Var) for value in (x, cos, sin)):
        return None
    if jt.flags.amp_reg or _output_requires_grad(x, cos, sin):
        return None
    dtype = _jittor_dtype_name(x.dtype)
    if dtype not in ("float32", "float16", "bfloat16"):
        return None
    shape = tuple(x.shape)
    if not shape or not x.numel() or shape[-1] % 2:
        return None
    storage = _interleaved_rope_storage(x)
    if storage is None:
        return None
    strides = []
    for freqs in (cos, sin):
        if (_jittor_dtype_name(freqs.dtype) != "float32"
                or not freqs._storage_is_contiguous() or freqs.ndim > x.ndim):
            return None
        dims = (1,) * (x.ndim - freqs.ndim) + tuple(freqs.shape)
        if dims[-1] != shape[-1] or any(a not in (1, b) for a, b in zip(dims, shape)):
            return None
        stride = 1
        freq_strides = []
        for dim in reversed(dims):
            freq_strides.append(stride if dim != 1 else 0)
            stride *= dim
        strides.append(tuple(reversed(freq_strides)))
    return storage, tuple(strides)


def _interleaved_offset(shape, strides, offset=0):
    # Compute one broadcast offset per pair, skipping axes whose frequencies
    # are shared. Lanes always read and write adjacent elements within a row.
    terms = ["col"]
    if offset:
        terms.append(str(offset))
    row_stride = 1
    for extent, stride in reversed(tuple(zip(shape[:-1], strides[:-1]))):
        if extent > 1 and stride:
            terms.append(f"(row / {row_stride} % {extent}) * {stride}")
        row_stride *= extent
    return " + ".join(terms)


@lru_cache(maxsize=128)
def _interleaved_rope_source(shape, x_strides, x_offset, freq_strides):
    x_index = _interleaved_offset(shape, x_strides, x_offset) if x_strides else "i"
    cos_index, sin_index = (_interleaved_offset(shape, strides) for strides in freq_strides)
    paired = x_offset % 2 == 0 and all(stride % 2 == 0 for stride in x_strides[:-1])
    header = r"""
    namespace jittor {
    template <typename T>
    struct alignas(2 * sizeof(T)) RopePair {
        T first, second;
    };

    template <bool aligned, typename T>
    __device__ __forceinline__ RopePair<T> rope_load(const T* ptr, index_t i) {
        if (aligned)
            return reinterpret_cast<const RopePair<T>*>(ptr)[i / 2];
        return {ptr[i], ptr[i + 1]};
    }

    template <bool aligned>
    __global__ static void interleaved_rope(
            const in0_type* __restrict__ x, const float* __restrict__ cos,
            const float* __restrict__ sin, out0_type* __restrict__ y, index_t total) {
        index_t stride = (index_t)gridDim.x * blockDim.x;
        for (index_t pair = (index_t)blockIdx.x * blockDim.x + threadIdx.x;
             pair < total; pair += stride) {
            index_t i = pair * 2;
            index_t row = i / %(dim)d;
            index_t col = i %% %(dim)d;
            index_t xi = %(x_index)s;
            index_t ci = %(cos_index)s;
            index_t si = %(sin_index)s;
            auto xy = rope_load<aligned>(x, xi);
            auto c = rope_load<aligned>(cos, ci);
            auto s = rope_load<aligned>(sin, si);
            float real = static_cast<float>(xy.first);
            float imag = static_cast<float>(xy.second);
            RopePair<out0_type> out{
                out0_type(__fadd_rn(__fmul_rn(real, c.first), __fmul_rn(-imag, s.first))),
                out0_type(__fadd_rn(__fmul_rn(imag, c.second), __fmul_rn(real, s.second))),
            };
            if (aligned) {
                reinterpret_cast<RopePair<out0_type>*>(y)[pair] = out;
            } else {
                y[i] = out.first;
                y[i + 1] = out.second;
            }
        }
    }
    }
    """ % {"dim": shape[-1], "x_index": x_index, "cos_index": cos_index, "sin_index": sin_index}
    launch = r"""
    index_t total = out0->num / 2;
    int blocks = std::min((index_t)65535, (total + 255) / 256);
    if (%(paired)d && ((((size_t)in0_p | (size_t)in1_p | (size_t)in2_p | (size_t)out0_p) & 7u) == 0)) {
        interleaved_rope<true><<<blocks, 256>>>(in0_p, in1_p, in2_p, out0_p, total);
    } else {
        interleaved_rope<false><<<blocks, 256>>>(in0_p, in1_p, in2_p, out0_p, total);
    }
    CHECK(0 == cudaGetLastError());
    """ % {"paired": paired}
    return header, launch


@optional_kernel("nn.interleaved_rope.inference", "cuda")
def _interleaved_rope_cuda(x, cos, sin):
    """Apply adjacent-pair rotation with FP32 arithmetic and one output rounding."""
    contract = _interleaved_rope_contract(x, cos, sin)
    if contract is None:
        return None
    (dense, strides, offset), freq_strides = contract
    header, launch = _interleaved_rope_source(tuple(x.shape), strides, offset, freq_strides)
    return jt.code(x.shape, x.dtype, [dense, cos, sin], cuda_header=header, cuda_src=launch).stop_grad()


__all__ = ["partial_rotary_embedding_cuda"]
