"""CUDA inference kernels for rotary position embeddings."""

import jittor as jt
from jittor._runtime.core_api import _output_requires_grad, _stop_grad_outputs

from ._cuda_inference import cached_source, device_index, on_acl


def _rotary_embedding_cuda(
    positions,
    q,
    k,
    cos_sin_cache,
    *,
    head_size,
    rotary_dim,
    is_neox_style,
):
    """Inference-only CUDA RoPE for GQA query/key shapes."""
    tensors = (positions, q, k, cos_sin_cache)
    if not all(isinstance(value, jt.Var) for value in tensors):
        return None
    if not (jt.flags.use_cuda and not _output_requires_grad(tensors)):
        return None
    if on_acl() or not is_neox_style:
        return None
    try:
        q_shape = tuple(int(size) for size in q.shape)
        k_shape = tuple(int(size) for size in k.shape)
        cache_shape = tuple(int(size) for size in cos_sin_cache.shape)
        token_count = int(positions.numel())
        head_size = int(head_size)
        rotary_dim = int(rotary_dim)
        devices = tuple(device_index(value) for value in tensors)
    except Exception:
        return None
    if any(device < 0 for device in devices) or len(set(devices)) != 1:
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
    value_dtypes = {str(value.dtype) for value in (q, k, cos_sin_cache)}
    if len(value_dtypes) != 1 or value_dtypes.pop() not in (
        "float16", "bfloat16", "float32"
    ):
        return None
    if str(positions.dtype) not in ("int32", "int64"):
        return None

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


def partial_rotary_embedding_cuda(q, k, cos, sin, *, prefix_tokens, rotary_dim=None):
    """Rotate the final tokens in ``q`` and ``k`` after an explicit prefix.

    The token and channel axes are the final two axes. ``cos`` and ``sin`` are
    shaped ``(rotated_tokens, rotary_dim)``. Channels after ``rotary_dim`` are
    copied unchanged. The inference-only kernel returns ``None`` when the input
    contract is not supported.
    """
    tensors = (q, k, cos, sin)
    if not all(isinstance(value, jt.Var) for value in tensors):
        return None
    if not (jt.flags.use_cuda and not _output_requires_grad(tensors)):
        return None
    if on_acl():
        return None
    dtypes = tuple(str(value.dtype) for value in tensors)
    if len(set(dtypes)) != 1 or dtypes[0] != "float32":
        return None
    try:
        devices = tuple(device_index(value) for value in tensors)
        q_shape = tuple(int(size) for size in q.shape)
        k_shape = tuple(int(size) for size in k.shape)
        cos_shape = tuple(int(size) for size in cos.shape)
        sin_shape = tuple(int(size) for size in sin.shape)
        prefix_count = int(prefix_tokens)
    except Exception:
        return None
    if any(device < 0 for device in devices) or len(set(devices)) != 1:
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
