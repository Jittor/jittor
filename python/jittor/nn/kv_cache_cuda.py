"""Private CUDA inference kernels for paged KV caches."""

import jittor as jt
from jittor._runtime.core_api import _output_requires_grad, _stop_grad_outputs

from ._cuda_inference import cached_source, device_index, on_acl


def _reshape_and_cache_cuda(key, value, kv_cache, slot_mapping):
    """Scatter ``key`` and ``value`` into a V1 paged KV cache in place."""
    tensors = (key, value, kv_cache, slot_mapping)
    if not all(isinstance(tensor, jt.Var) for tensor in tensors):
        return None
    if not (jt.flags.use_cuda and not _output_requires_grad(
            key, value, kv_cache, slot_mapping)):
        return None
    if on_acl():
        return None
    try:
        key_shape = tuple(int(size) for size in key.shape)
        value_shape = tuple(int(size) for size in value.shape)
        cache_shape = tuple(int(size) for size in kv_cache.shape)
        devices = tuple(device_index(tensor) for tensor in tensors)
    except Exception:
        return None
    if any(device < 0 for device in devices) or len(set(devices)) != 1:
        return None
    if len(key_shape) != 3 or value_shape != key_shape or len(cache_shape) != 5:
        return None
    if cache_shape[1] != 2 or cache_shape[3:] != key_shape[1:]:
        return None
    if int(slot_mapping.numel()) < key_shape[0]:
        return None
    value_dtypes = {str(tensor.dtype) for tensor in (key, value, kv_cache)}
    if len(value_dtypes) != 1 or value_dtypes.pop() not in (
        "float16", "bfloat16", "float32"
    ):
        return None
    if str(slot_mapping.dtype) not in ("int32", "int64"):
        return None

    cuda_src = cached_source(r"""
    __global__ static void reshape_and_cache(
            const in0_type* key, const in1_type* value,
            const in2_type* slots, out0_type* cache, int64_t total) {
        int64_t index = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
        int64_t stride = (int64_t)blockDim.x * gridDim.x;
        for (; index < total; index += stride) {
            int64_t token = index / %(token_stride)d;
            int slot = (int)slots[token];
            if (slot < 0 || slot >= %(slot_capacity)d) continue;
            int block = slot / %(block_size)d;
            int offset = slot %% %(block_size)d;
            int64_t within_token = index - token * %(token_stride)d;
            int64_t key_index =
                (((int64_t)block * 2 * %(block_size)d + offset)
                 * %(token_stride)d) + within_token;
            int64_t value_index = key_index +
                (int64_t)%(block_size)d * %(token_stride)d;
            cache[key_index] = out0_type(key[index]);
            cache[value_index] = out0_type(value[index]);
        }
    }
    int64_t total = in0->num;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    if (blocks > 4096) blocks = 4096;
    if (total) reshape_and_cache<<<blocks, threads>>>(
        in0_p, in1_p, in2_p, out0_p, total);
    CHECK(0 == cudaGetLastError());
    """, {
        "block_size": cache_shape[2],
        "slot_capacity": cache_shape[0] * cache_shape[2],
        "token_stride": key_shape[1] * key_shape[2],
    })
    jt.code([key, value, slot_mapping], [kv_cache], cuda_src=cuda_src)
    return _stop_grad_outputs(kv_cache)


def _paged_attention_decode_cuda(
    query,
    kv_cache,
    seq_lens,
    block_table,
    softmax_scale,
):
    """Decode one token per request from a V1 paged KV cache."""
    tensors = (query, kv_cache, seq_lens, block_table)
    if not all(isinstance(tensor, jt.Var) for tensor in tensors):
        return None
    if not (jt.flags.use_cuda and not _output_requires_grad(tensors)):
        return None
    if on_acl():
        return None
    try:
        query_shape = tuple(int(size) for size in query.shape)
        cache_shape = tuple(int(size) for size in kv_cache.shape)
        table_shape = tuple(int(size) for size in block_table.shape)
        devices = tuple(device_index(tensor) for tensor in tensors)
        scale = float(softmax_scale)
    except Exception:
        return None
    if any(device < 0 for device in devices) or len(set(devices)) != 1:
        return None
    if len(query_shape) != 3 or len(cache_shape) != 5 or len(table_shape) != 2:
        return None
    requests, query_heads, head_dim = query_shape
    if cache_shape[1] != 2 or cache_shape[-1] != head_dim:
        return None
    kv_heads = cache_shape[3]
    if (
        requests <= 0
        or query_heads <= 0
        or kv_heads <= 0
        or query_heads % kv_heads
        or head_dim <= 0
        or head_dim > 256
    ):
        return None
    if table_shape[0] != requests or int(seq_lens.numel()) < requests:
        return None
    if str(query.dtype) != str(kv_cache.dtype) or str(query.dtype) not in (
        "float16", "bfloat16", "float32"
    ):
        return None
    if str(seq_lens.dtype) not in ("int32", "int64"):
        return None
    if str(block_table.dtype) not in ("int32", "int64"):
        return None
    if not (scale > 0 and scale < float("inf")):
        return None

    threads = 32
    while threads < head_dim:
        threads *= 2
    warps = threads // 32
    cuda_src = cached_source(r"""
    __device__ __forceinline__ float warp_sum(float value) {
        for (int offset = 16; offset > 0; offset >>= 1)
            value += __shfl_down_sync(0xffffffff, value, offset);
        return value;
    }
    __global__ static void paged_attention_decode(
            const in0_type* query, const in1_type* cache,
            const in2_type* seq_lens, const in3_type* block_table,
            out0_type* output) {
        int request = blockIdx.x / %(query_heads)d;
        int query_head = blockIdx.x %% %(query_heads)d;
        int kv_head = query_head / %(head_repeat)d;
        int tid = threadIdx.x;
        int lane = tid & 31;
        int warp = tid >> 5;
        __shared__ float warp_buf[%(warps)d];
        __shared__ float score;
        __shared__ float running_max;
        __shared__ float denominator;
        __shared__ float alpha;
        __shared__ float beta;
        if (tid == 0) {
            running_max = -1.0e30f;
            denominator = 0.0f;
        }
        __syncthreads();

        float accumulator = 0.0f;
        int seq_len = (int)seq_lens[request];
        int64_t query_base =
            ((int64_t)request * %(query_heads)d + query_head) * %(head_dim)d;
        for (int position = 0; position < seq_len; ++position) {
            int logical_block = position / %(block_size)d;
            int offset = position %% %(block_size)d;
            int physical_block = (int)block_table[
                request * %(max_blocks)d + logical_block];
            int64_t key_base =
                ((((int64_t)physical_block * 2) * %(block_size)d + offset)
                 * %(kv_heads)d + kv_head) * %(head_dim)d;
            float partial = 0.0f;
            for (int dim = tid; dim < %(head_dim)d; dim += blockDim.x)
                partial += static_cast<float>(query[query_base + dim])
                         * static_cast<float>(cache[key_base + dim]);
            partial = warp_sum(partial);
            if (lane == 0) warp_buf[warp] = partial;
            __syncthreads();
            if (warp == 0) {
                float total = lane < %(warps)d ? warp_buf[lane] : 0.0f;
                total = warp_sum(total);
                if (lane == 0) score = total * %(scale).9gf;
            }
            __syncthreads();
            if (tid == 0) {
                float next_max = fmaxf(running_max, score);
                alpha = expf(running_max - next_max);
                beta = expf(score - next_max);
                denominator = denominator * alpha + beta;
                running_max = next_max;
            }
            __syncthreads();
            int64_t value_base = key_base +
                (int64_t)%(block_size)d * %(kv_heads)d * %(head_dim)d;
            if (tid < %(head_dim)d)
                accumulator = accumulator * alpha
                    + beta * static_cast<float>(cache[value_base + tid]);
            __syncthreads();
        }
        if (tid < %(head_dim)d)
            output[query_base + tid] = out0_type(accumulator / denominator);
    }
    paged_attention_decode<<<%(blocks)d, %(threads)d>>>(
        in0_p, in1_p, in2_p, in3_p, out0_p);
    CHECK(0 == cudaGetLastError());
    """, {
        "blocks": requests * query_heads,
        "block_size": cache_shape[2],
        "head_dim": head_dim,
        "head_repeat": query_heads // kv_heads,
        "kv_heads": kv_heads,
        "max_blocks": table_shape[1],
        "query_heads": query_heads,
        "scale": scale,
        "threads": threads,
        "warps": warps,
    })
    return _stop_grad_outputs(
        jt.code(query.shape, query.dtype, tensors, cuda_src=cuda_src))


__all__ = []
