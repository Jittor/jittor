"""CUDA inference kernels for partial rotary position embeddings."""

import jittor as jt


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
    if not (jt.flags.use_cuda and getattr(jt.flags, "no_grad", 0)):
        return None
    if getattr(jt.compiler, "has_acl", 0):
        return None
    dtypes = tuple(str(value.dtype) for value in tensors)
    if len(set(dtypes)) != 1 or dtypes[0] != "float32":
        return None
    try:
        devices = tuple(int(value.get_device()) for value in tensors)
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

    cuda_src = r"""
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
    """ % {
        "head_dim": head_dim,
        "token_count": token_count,
        "prefix_count": prefix_count,
        "rotate": rotate,
        "table_dim": table_dim,
    }
    return jt.code(
        [q.shape, k.shape],
        [q.dtype, k.dtype],
        [q, k, cos, sin],
        cuda_src=cuda_src,
    )


__all__ = ["partial_rotary_embedding_cuda"]
