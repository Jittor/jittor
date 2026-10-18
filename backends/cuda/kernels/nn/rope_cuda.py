"""CUDA inference kernels for rotary position embeddings."""
import math

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
                    float sign = dim < half ? -1.0f : 1.0f;
                    float cos_value = static_cast<float>(
                        cache[position * %(cache_width)d + frequency_dim]);
                    float sin_value = static_cast<float>(
                        cache[position * %(cache_width)d + half + frequency_dim]);
                    out_q[index] = out0_type(static_cast<float>(q[index]) * cos_value
                        + sign * static_cast<float>(q[other_index]) * sin_value);
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
                    float sign = dim < half ? -1.0f : 1.0f;
                    float cos_value = static_cast<float>(
                        cache[position * %(cache_width)d + frequency_dim]);
                    float sin_value = static_cast<float>(
                        cache[position * %(cache_width)d + half + frequency_dim]);
                    out_k[index] = out1_type(static_cast<float>(k[index]) * cos_value
                        + sign * static_cast<float>(k[other_index]) * sin_value);
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


def _qk_rms_norm_rotary_contract(
    positions,
    q,
    k,
    q_weight,
    k_weight,
    cos_sin_cache,
    *,
    head_size,
    rotary_dim,
    is_neox_style,
    epsilon,
):
    tensors = (positions, q, k, q_weight, k_weight, cos_sin_cache)
    if not all(isinstance(value, jt.Var) for value in tensors):
        return None
    if _output_requires_grad(tensors) or not is_neox_style:
        return None
    try:
        q_shape = tuple(int(size) for size in q.shape)
        k_shape = tuple(int(size) for size in k.shape)
        q_weight_shape = tuple(int(size) for size in q_weight.shape)
        k_weight_shape = tuple(int(size) for size in k_weight.shape)
        cache_shape = tuple(int(size) for size in cos_sin_cache.shape)
        token_count = int(positions.numel())
        head_size = int(head_size)
        rotary_dim = int(rotary_dim)
        epsilon = float(epsilon)
    except Exception:
        return None
    if len(q_shape) != 2 or len(k_shape) != 2 or len(cache_shape) != 2:
        return None
    if q_shape[0] != token_count or k_shape[0] != token_count:
        return None
    if head_size <= 0 or head_size > 4096:
        return None
    if q_shape[1] % head_size or k_shape[1] % head_size:
        return None
    if q_weight_shape != (head_size,) or k_weight_shape != (head_size,):
        return None
    if rotary_dim <= 0 or rotary_dim > head_size or rotary_dim % 2:
        return None
    if cache_shape[0] <= 0 or cache_shape[1] != rotary_dim:
        return None
    value_dtypes = {
        _jittor_dtype_name(value.dtype)
        for value in (q, k, q_weight, k_weight, cos_sin_cache)
    }
    if len(value_dtypes) != 1 or value_dtypes.pop() not in (
        "float16", "bfloat16", "float32"
    ):
        return None
    if _jittor_dtype_name(positions.dtype) not in ("int32", "int64"):
        return None
    if not math.isfinite(epsilon) or epsilon <= 0:
        return None
    return q_shape, k_shape, cache_shape, head_size, rotary_dim, epsilon


def _qk_rms_norm_rotary_supported(*args, **kwargs):
    return _qk_rms_norm_rotary_contract(*args, **kwargs) is not None


@optional_kernel(
    "nn.qk_rms_norm_rotary.inference",
    ("cuda", "rocm_legacy", "corex_legacy"),
    supports=_qk_rms_norm_rotary_supported,
)
def _qk_rms_norm_rotary_cuda(
    positions,
    q,
    k,
    q_weight,
    k_weight,
    cos_sin_cache,
    *,
    head_size,
    rotary_dim,
    is_neox_style,
    epsilon,
):
    """Fuse per-head Q/K RMSNorm and NeoX RoPE in one CUDA launch."""
    q_shape, k_shape, cache_shape, head_size, rotary_dim, epsilon = (
        _qk_rms_norm_rotary_contract(
            positions,
            q,
            k,
            q_weight,
            k_weight,
            cos_sin_cache,
            head_size=head_size,
            rotary_dim=rotary_dim,
            is_neox_style=is_neox_style,
            epsilon=epsilon,
        )
    )
    threads = 32
    while threads < min(head_size, 1024):
        threads *= 2
    warps = threads // 32
    cuda_src = cached_source(r"""
    __device__ __forceinline__ float qk_warp_sum(float value) {
        for (int offset = 16; offset > 0; offset >>= 1)
            value += __shfl_down_sync(0xffffffff, value, offset);
        return value;
    }
    __global__ static void qk_rms_norm_rotary(
            const in0_type* positions,
            const in1_type* q, const in2_type* k,
            const in3_type* q_weight, const in4_type* k_weight,
            const in5_type* cache,
            out0_type* out_q, out1_type* out_k,
            int q_rows, int q_heads, int k_heads) {
        int row = blockIdx.x;
        bool use_q = row < q_rows;
        int local_row = use_q ? row : row - q_rows;
        int heads = use_q ? q_heads : k_heads;
        int token = local_row / heads;
        int tid = threadIdx.x;
        int lane = tid & 31;
        int warp = tid >> 5;
        int64_t base = (int64_t)local_row * %(head_size)d;
        __shared__ float warp_sums[%(warps)d];
        __shared__ float inverse_rms;
        float sum = 0.0f;
        for (int dim = tid; dim < %(head_size)d; dim += blockDim.x) {
            float value = use_q ? static_cast<float>(q[base + dim])
                                : static_cast<float>(k[base + dim]);
            sum += value * value;
        }
        sum = qk_warp_sum(sum);
        if (lane == 0) warp_sums[warp] = sum;
        __syncthreads();
        if (warp == 0) {
            float total = lane < %(warps)d ? warp_sums[lane] : 0.0f;
            total = qk_warp_sum(total);
            if (lane == 0)
                inverse_rms = rsqrtf(
                    total / %(head_size)d.0f + %(epsilon).9gf);
        }
        __syncthreads();
        int position = (int)positions[token];
        int half = %(rotary_dim)d / 2;
        for (int dim = tid; dim < %(head_size)d; dim += blockDim.x) {
            float value = use_q ? static_cast<float>(q[base + dim])
                                : static_cast<float>(k[base + dim]);
            float weight = use_q ? static_cast<float>(q_weight[dim])
                                 : static_cast<float>(k_weight[dim]);
            float result = value * inverse_rms * weight;
            if (dim < %(rotary_dim)d) {
                int other_dim = dim < half ? dim + half : dim - half;
                float other_value = use_q
                    ? static_cast<float>(q[base + other_dim])
                    : static_cast<float>(k[base + other_dim]);
                float other_weight = use_q
                    ? static_cast<float>(q_weight[other_dim])
                    : static_cast<float>(k_weight[other_dim]);
                int frequency_dim = dim < half ? dim : dim - half;
                float sign = dim < half ? -1.0f : 1.0f;
                float cos_value = static_cast<float>(
                    cache[position * %(cache_width)d + frequency_dim]);
                float sin_value = static_cast<float>(
                    cache[position * %(cache_width)d + half + frequency_dim]);
                result = result * cos_value + sign
                    * other_value * inverse_rms * other_weight * sin_value;
            }
            if (use_q) out_q[base + dim] = out0_type(result);
            else out_k[base + dim] = out1_type(result);
        }
    }
    int q_heads = %(q_width)d / %(head_size)d;
    int k_heads = %(k_width)d / %(head_size)d;
    int q_rows = in1->num / %(head_size)d;
    int total_rows = q_rows + in2->num / %(head_size)d;
    if (total_rows) qk_rms_norm_rotary<<<total_rows, %(threads)d>>>(
        in0_p, in1_p, in2_p, in3_p, in4_p, in5_p,
        out0_p, out1_p, q_rows, q_heads, k_heads);
    CHECK(0 == cudaGetLastError());
    """, {
        "cache_width": cache_shape[1],
        "epsilon": epsilon,
        "head_size": head_size,
        "k_width": k_shape[1],
        "q_width": q_shape[1],
        "rotary_dim": rotary_dim,
        "threads": threads,
        "warps": warps,
    })
    return _stop_grad_outputs(jt.code(
        [q.shape, k.shape], [q.dtype, k.dtype],
        [positions, q, k, q_weight, k_weight, cos_sin_cache],
        cuda_src=cuda_src,
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


__all__ = ["partial_rotary_embedding_cuda"]
