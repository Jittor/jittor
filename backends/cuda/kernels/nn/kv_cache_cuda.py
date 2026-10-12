"""Private CUDA inference kernels for paged KV caches."""
from jittor._core.dtypes import dtype_name as _jittor_dtype_name

import jittor as jt
from jittor._runtime.core_api import _output_requires_grad, _stop_grad_outputs
from jittor._runtime.dispatch import optional_kernel

from ._inference import cached_source


@optional_kernel("nn.reshape_and_cache_cuda", ("cuda", "rocm_legacy", "corex_legacy"),
                 dtypes={"float16", "bfloat16", "float32", "int32", "int64"})
def _reshape_and_cache_cuda(key, value, kv_cache, slot_mapping):
    """Scatter ``key`` and ``value`` into a V1 paged KV cache in place."""
    tensors = (key, value, kv_cache, slot_mapping)
    if not all(isinstance(tensor, jt.Var) for tensor in tensors):
        return None
    if _output_requires_grad(key, value, kv_cache, slot_mapping):
        return None
    try:
        key_shape = tuple(int(size) for size in key.shape)
        value_shape = tuple(int(size) for size in value.shape)
        cache_shape = tuple(int(size) for size in kv_cache.shape)
    except Exception:
        return None
    if len(key_shape) != 3 or value_shape != key_shape or len(cache_shape) != 5:
        return None
    if cache_shape[1] != 2 or cache_shape[3:] != key_shape[1:]:
        return None
    if int(slot_mapping.numel()) < key_shape[0]:
        return None
    value_dtypes = {_jittor_dtype_name(tensor.dtype) for tensor in (key, value, kv_cache)}
    if len(value_dtypes) != 1 or value_dtypes.pop() not in (
        "float16", "bfloat16", "float32"
    ):
        return None
    if _jittor_dtype_name(slot_mapping.dtype) not in ("int32", "int64"):
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


@optional_kernel("nn.paged_attention_decode_cuda", ("cuda", "rocm_legacy", "corex_legacy"),
                 dtypes={"float16", "bfloat16", "float32", "int32", "int64"})
def _paged_attention_decode_cuda(
    query,
    kv_cache,
    seq_lens,
    block_table,
    softmax_scale,
    out=None,
):
    """Decode one token per request from a V1 paged KV cache."""
    tensors = (query, kv_cache, seq_lens, block_table)
    if not all(isinstance(tensor, jt.Var) for tensor in tensors):
        return None
    if out is not None and not isinstance(out, jt.Var):
        return None
    if _output_requires_grad(tensors if out is None else (tensors, out)):
        return None
    try:
        query_shape = tuple(int(size) for size in query.shape)
        cache_shape = tuple(int(size) for size in kv_cache.shape)
        table_shape = tuple(int(size) for size in block_table.shape)
        scale = float(softmax_scale)
    except Exception:
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
    if _jittor_dtype_name(query.dtype) != _jittor_dtype_name(kv_cache.dtype) or _jittor_dtype_name(query.dtype) not in (
        "float16", "bfloat16", "float32"
    ):
        return None
    if _jittor_dtype_name(seq_lens.dtype) not in ("int32", "int64"):
        return None
    if _jittor_dtype_name(block_table.dtype) not in ("int32", "int64"):
        return None
    if out is not None and (
        int(out.numel()) != int(query.numel())
        or _jittor_dtype_name(out.dtype) != _jittor_dtype_name(query.dtype)
    ):
        return None
    if not (scale > 0 and scale < float("inf")):
        return None
    # Give independent warps different key positions.  The previous kernel
    # made the whole block walk the sequence serially and paid four block-wide
    # barriers per token.  Decode has only one query token, so position-level
    # parallelism is the dimension that keeps the SM busy as the cache grows.
    warps = 8
    threads = warps * 32
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
        __shared__ float warp_max[%(warps)d];
        __shared__ float warp_denominator[%(warps)d];
        __shared__ float warp_output[%(warps)d][%(head_dim)d];

        int seq_len = (int)seq_lens[request];
        int64_t query_base =
            ((int64_t)request * %(query_heads)d + query_head) * %(head_dim)d;
        float local_max = -1.0e30f;
        float local_denominator = 0.0f;
        float accumulator[%(values_per_lane)d];
        #pragma unroll
        for (int item = 0; item < %(values_per_lane)d; ++item)
            accumulator[item] = 0.0f;

        for (int position = warp; position < seq_len; position += %(warps)d) {
            int logical_block = position / %(block_size)d;
            int offset = position %% %(block_size)d;
            int physical_block = (int)block_table[
                request * %(max_blocks)d + logical_block];
            int64_t key_base =
                ((((int64_t)physical_block * 2) * %(block_size)d + offset)
                 * %(kv_heads)d + kv_head) * %(head_dim)d;
            float partial = 0.0f;
            // Every warp needs the complete dot product.  Use lane-strided
            // dimensions rather than block-strided ones because warps now own
            // independent cache positions.
            for (int dim = lane; dim < %(head_dim)d; dim += 32)
                partial += static_cast<float>(query[query_base + dim])
                         * static_cast<float>(cache[key_base + dim]);
            partial = warp_sum(partial);
            float score = __shfl_sync(0xffffffff, partial, 0) * %(scale).9gf;
            float next_max = fmaxf(local_max, score);
            float alpha = expf(local_max - next_max);
            float beta = expf(score - next_max);
            local_denominator = local_denominator * alpha + beta;
            int64_t value_base = key_base +
                (int64_t)%(block_size)d * %(kv_heads)d * %(head_dim)d;
            #pragma unroll
            for (int item = 0; item < %(values_per_lane)d; ++item) {
                int dim = lane + item * 32;
                if (dim < %(head_dim)d)
                    accumulator[item] = accumulator[item] * alpha
                        + beta * static_cast<float>(cache[value_base + dim]);
            }
            local_max = next_max;
        }

        if (lane == 0) {
            warp_max[warp] = local_max;
            warp_denominator[warp] = local_denominator;
        }
        #pragma unroll
        for (int item = 0; item < %(values_per_lane)d; ++item) {
            int dim = lane + item * 32;
            if (dim < %(head_dim)d)
                warp_output[warp][dim] = accumulator[item];
        }
        __syncthreads();

        if (tid < %(head_dim)d) {
            if (seq_len <= 0) {
                output[query_base + tid] = out0_type(0.0f);
                return;
            }
            float global_max = warp_max[0];
            #pragma unroll
            for (int owner = 1; owner < %(warps)d; ++owner)
                global_max = fmaxf(global_max, warp_max[owner]);
            float denominator = 0.0f;
            float result = 0.0f;
            #pragma unroll
            for (int owner = 0; owner < %(warps)d; ++owner) {
                float weight = expf(warp_max[owner] - global_max);
                denominator += warp_denominator[owner] * weight;
                result += warp_output[owner][tid] * weight;
            }
            output[query_base + tid] = out0_type(result / denominator);
        }
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
        "values_per_lane": (head_dim + 31) // 32,
        "warps": warps,
    })
    if out is None:
        out = jt.code(query.shape, query.dtype, tensors, cuda_src=cuda_src)
    else:
        # vLLM owns and reuses its attention output holder.  Let the decode
        # kernel fill that storage directly instead of allocating a temporary
        # and launching a setitem kernel to copy it afterwards.
        jt.code(list(tensors), [out], cuda_src=cuda_src)
    return _stop_grad_outputs(out)


@optional_kernel("nn.reshape_cache_paged_attention_decode_cuda",
                 ("cuda", "rocm_legacy", "corex_legacy"),
                 dtypes={"float16", "bfloat16", "float32", "int32", "int64"})
def _reshape_cache_and_paged_attention_decode_cuda(
    query,
    key,
    value,
    kv_cache,
    slot_mapping,
    seq_lens,
    block_table,
    softmax_scale,
    out,
):
    """Fuse the decode-step KV scatter and paged attention in one launch."""
    tensors = (query, key, value, kv_cache, slot_mapping, seq_lens, block_table)
    if not all(isinstance(tensor, jt.Var) for tensor in tensors):
        return None
    if not isinstance(out, jt.Var) or _output_requires_grad(tensors + (out,)):
        return None
    try:
        query_shape = tuple(int(size) for size in query.shape)
        key_shape = tuple(int(size) for size in key.shape)
        value_shape = tuple(int(size) for size in value.shape)
        cache_shape = tuple(int(size) for size in kv_cache.shape)
        table_shape = tuple(int(size) for size in block_table.shape)
        scale = float(softmax_scale)
    except Exception:
        return None
    if (len(query_shape) != 3 or len(key_shape) != 3
            or value_shape != key_shape or len(cache_shape) != 5
            or len(table_shape) != 2):
        return None
    requests, query_heads, head_dim = query_shape
    kv_heads = cache_shape[3]
    if (cache_shape[1] != 2 or cache_shape[-1] != head_dim
            or key_shape != (requests, kv_heads, head_dim)
            or table_shape[0] != requests or requests <= 0
            or query_heads <= 0 or kv_heads <= 0
            or query_heads % kv_heads or head_dim <= 0 or head_dim > 256
            or int(slot_mapping.numel()) < requests
            or int(seq_lens.numel()) < requests):
        return None
    value_dtype = _jittor_dtype_name(query.dtype)
    if (value_dtype not in ("float16", "bfloat16", "float32")
            or any(_jittor_dtype_name(tensor.dtype) != value_dtype
                   for tensor in (key, value, kv_cache, out))):
        return None
    if (_jittor_dtype_name(slot_mapping.dtype) not in ("int32", "int64")
            or _jittor_dtype_name(seq_lens.dtype) not in ("int32", "int64")
            or _jittor_dtype_name(block_table.dtype) not in ("int32", "int64")
            or int(out.numel()) != int(query.numel())
            or not (scale > 0 and scale < float("inf"))):
        return None

    warps = 8
    threads = warps * 32
    cuda_src = cached_source(r"""
    __device__ __forceinline__ float warp_sum(float value) {
        for (int offset = 16; offset > 0; offset >>= 1)
            value += __shfl_down_sync(0xffffffff, value, offset);
        return value;
    }
    __global__ static void reshape_cache_paged_attention_decode(
            const in0_type* query, const in1_type* key,
            const in2_type* value, const in3_type* slot_mapping,
            const in4_type* seq_lens, const in5_type* block_table,
            out0_type* cache, out1_type* output) {
        int tid = threadIdx.x;
        if (blockIdx.x >= %(attention_blocks)d) {
            int64_t index =
                (int64_t)(blockIdx.x - %(attention_blocks)d) * blockDim.x + tid;
            if (index < %(write_elements)d) {
                int request = index / %(token_stride)d;
                int within_token = index - (int64_t)request * %(token_stride)d;
                int slot = (int)slot_mapping[request];
                if (slot >= 0 && slot < %(slot_capacity)d) {
                    int current_block = slot / %(block_size)d;
                    int current_offset = slot %% %(block_size)d;
                    int64_t key_base =
                        (((int64_t)current_block * 2 * %(block_size)d
                          + current_offset) * %(token_stride)d)
                        + within_token;
                    int64_t value_base = key_base
                        + (int64_t)%(block_size)d * %(token_stride)d;
                    cache[key_base] = out0_type(key[index]);
                    cache[value_base] = out0_type(value[index]);
                }
            }
            return;
        }
        int request = blockIdx.x / %(query_heads)d;
        int query_head = blockIdx.x %% %(query_heads)d;
        int kv_head = query_head / %(head_repeat)d;
        int lane = tid & 31;
        int warp = tid >> 5;

        __shared__ float warp_max[%(warps)d];
        __shared__ float warp_denominator[%(warps)d];
        __shared__ float warp_output[%(warps)d][%(head_dim)d];
        int seq_len = (int)seq_lens[request];
        int64_t query_base =
            ((int64_t)request * %(query_heads)d + query_head) * %(head_dim)d;
        float local_max = -1.0e30f;
        float local_denominator = 0.0f;
        float accumulator[%(values_per_lane)d];
        #pragma unroll
        for (int item = 0; item < %(values_per_lane)d; ++item)
            accumulator[item] = 0.0f;

        for (int position = warp; position < seq_len; position += %(warps)d) {
            int logical_block = position / %(block_size)d;
            int offset = position %% %(block_size)d;
            int physical_block = (int)block_table[
                request * %(max_blocks)d + logical_block];
            int64_t key_base =
                ((((int64_t)physical_block * 2) * %(block_size)d + offset)
                 * %(kv_heads)d + kv_head) * %(head_dim)d;
            bool is_current = position == seq_len - 1;
            int64_t current_base =
                ((int64_t)request * %(kv_heads)d + kv_head) * %(head_dim)d;
            float partial = 0.0f;
            for (int dim = lane; dim < %(head_dim)d; dim += 32) {
                float key_value = is_current
                    ? static_cast<float>(key[current_base + dim])
                    : static_cast<float>(cache[key_base + dim]);
                partial += static_cast<float>(query[query_base + dim])
                         * key_value;
            }
            partial = warp_sum(partial);
            float score = __shfl_sync(0xffffffff, partial, 0) * %(scale).9gf;
            float next_max = fmaxf(local_max, score);
            float alpha = expf(local_max - next_max);
            float beta = expf(score - next_max);
            local_denominator = local_denominator * alpha + beta;
            int64_t value_base = key_base
                + (int64_t)%(block_size)d * %(kv_heads)d * %(head_dim)d;
            #pragma unroll
            for (int item = 0; item < %(values_per_lane)d; ++item) {
                int dim = lane + item * 32;
                if (dim < %(head_dim)d) {
                    float value_value = is_current
                        ? static_cast<float>(value[current_base + dim])
                        : static_cast<float>(cache[value_base + dim]);
                    accumulator[item] = accumulator[item] * alpha
                        + beta * value_value;
                }
            }
            local_max = next_max;
        }
        if (lane == 0) {
            warp_max[warp] = local_max;
            warp_denominator[warp] = local_denominator;
        }
        #pragma unroll
        for (int item = 0; item < %(values_per_lane)d; ++item) {
            int dim = lane + item * 32;
            if (dim < %(head_dim)d)
                warp_output[warp][dim] = accumulator[item];
        }
        __syncthreads();
        if (tid < %(head_dim)d) {
            if (seq_len <= 0) {
                output[query_base + tid] = out1_type(0.0f);
                return;
            }
            float global_max = warp_max[0];
            #pragma unroll
            for (int owner = 1; owner < %(warps)d; ++owner)
                global_max = fmaxf(global_max, warp_max[owner]);
            float denominator = 0.0f;
            float result = 0.0f;
            #pragma unroll
            for (int owner = 0; owner < %(warps)d; ++owner) {
                float weight = expf(warp_max[owner] - global_max);
                denominator += warp_denominator[owner] * weight;
                result += warp_output[owner][tid] * weight;
            }
            output[query_base + tid] = out1_type(result / denominator);
        }
    }
    reshape_cache_paged_attention_decode<<<%(blocks)d, %(threads)d>>>(
        in0_p, in1_p, in2_p, in3_p, in4_p, in5_p, out0_p, out1_p);
    CHECK(0 == cudaGetLastError());
    """, {
        "attention_blocks": requests * query_heads,
        "blocks": (requests * query_heads
                   + (requests * kv_heads * head_dim + threads - 1) // threads),
        "block_size": cache_shape[2],
        "head_dim": head_dim,
        "head_repeat": query_heads // kv_heads,
        "kv_heads": kv_heads,
        "max_blocks": table_shape[1],
        "query_heads": query_heads,
        "scale": scale,
        "slot_capacity": cache_shape[0] * cache_shape[2],
        "threads": threads,
        "token_stride": kv_heads * head_dim,
        "values_per_lane": (head_dim + 31) // 32,
        "warps": warps,
        "write_elements": requests * kv_heads * head_dim,
    })
    jt.code(
        [query, key, value, slot_mapping, seq_lens, block_table],
        [kv_cache, out], cuda_src=cuda_src)
    return _stop_grad_outputs(out)


__all__ = []
