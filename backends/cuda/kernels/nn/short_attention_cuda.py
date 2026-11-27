"""CUDA fusion of a short, explicitly staged FP16 attention readout.

Both matrix products accumulate in float32. Store-equivalent casts round the
scores, each bias add, and the softmax probabilities to FP16 before the next
stage. A fresh sequence-major output replaces the requested contiguous copy.
"""

from functools import lru_cache

import jittor as jt
from jittor._core.dtypes import dtype_name


def _offset_expression(layout, row, column):
    shape, strides, offset = layout
    terms = [f"{offset}LL"]
    span = 1
    for extent, stride in reversed(tuple(zip(shape[:-2], strides[:-2]))):
        if extent > 1 and stride:
            terms.append(f"(prefix / {span} % {extent}) * {stride}LL")
        span *= extent
    if strides[-2]:
        terms.append(f"({row}) * {strides[-2]}LL")
    if strides[-1]:
        terms.append(f"({column}) * {strides[-1]}LL")
    return " + ".join(terms)


def _tile_load(layout, name, tile, row_start, rows, tile_rows, width, pitch):
    offset = _offset_expression(layout, "row", "col")
    _, stride, start = layout
    single = f'''
    for (int item = threadIdx.x; item < {tile_rows * pitch}; item += 128) {{
        const int idx = item / {pitch};
        const int row = ({row_start}) + idx, col = item % {pitch};
        {tile}[swizzle(idx, col, {pitch})] = row < {rows} && col < {width} ?
            {name}[{offset}] : __float2half_rn(0.0f);
    }}'''
    if width % 8 or stride[-1] != 1 or start % 8 or any(step % 8 for step in stride[:-1]):
        return single
    return f'''
    if ((reinterpret_cast<uintptr_t>({name}) & 15) == 0) {{
        for (int item = threadIdx.x * 8; item < {tile_rows * pitch}; item += 128 * 8) {{
            const int idx = item / {pitch};
            const int row = ({row_start}) + idx, col = item % {pitch};
            const bool valid = row < {rows} && col < {width};
            copy_tile({tile} + swizzle(idx, col, {pitch}), {name} + (valid ? {offset} : 0), valid);
        }}
    }} else {{
        {single}
    }}'''


@lru_cache(maxsize=256)
def _attention_source(layouts):
    batch, heads, queries, depth = layouts[0][0]
    keys, width = layouts[2][0][-2:]
    padded_keys = 64
    padded_depth = max(16, 1 << (depth - 1).bit_length())
    padded_width = max(16, 1 << (width - 1).bit_length())
    add_count = len(layouts) - 3
    bias_args = ''.join(f', const in{index + 3}_type* bias{index}' for index in range(add_count))
    bias_loads = ''.join(
        f'                score = __hadd(score, bias{index}[{_offset_expression(bias, "row", "col")}]);\n'
        for index, bias in enumerate(layouts[3:]))
    query_load = _tile_load(layouts[0], 'query', 'q_tile', 'first', queries, 16, depth, padded_depth)
    key_load = _tile_load(layouts[1], 'key', 'k_tile', 0, keys, padded_keys, depth, padded_depth)
    value_load = _tile_load(layouts[2], 'value', 'v_tile', 0, keys, padded_keys, width, padded_width)
    launch_args = ''.join(f', in{index + 3}_p' for index in range(add_count))
    return f'''
__device__ __forceinline__ int swizzle(int row, int col, int pitch) {{
    const int shift = pitch == 64 ? 0 : (pitch == 32 ? 1 : 2);
    const int phase = (row >> shift) & (pitch / 8 - 1);
    return row * pitch + (col ^ (phase * 8));
}}
__device__ __forceinline__ void copy_tile(__half* dst, const __half* src, bool valid) {{
    const uint32_t offset = __cvta_generic_to_shared(dst);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;" : :
        "r"(offset), "l"(src), "r"(valid ? 16 : 0));
}}
__device__ __forceinline__ void load_a(const __half* address, uint32_t (&regs)[4]) {{
    uint32_t shared_address = __cvta_generic_to_shared(address);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{%0,%1,%2,%3}}, [%4];"
        : "=r"(regs[0]), "=r"(regs[1]), "=r"(regs[2]), "=r"(regs[3]) : "r"(shared_address));
}}
__device__ __forceinline__ void load_b(const __half* address, uint32_t (&regs)[2]) {{
    uint32_t shared_address = __cvta_generic_to_shared(address);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {{%0,%1}}, [%2];"
        : "=r"(regs[0]), "=r"(regs[1]) : "r"(shared_address));
}}
__device__ __forceinline__ void load_b_trans(const __half* address, uint32_t (&regs)[2]) {{
    uint32_t shared_address = __cvta_generic_to_shared(address);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {{%0,%1}}, [%2];"
        : "=r"(regs[0]), "=r"(regs[1]) : "r"(shared_address));
}}
__device__ __forceinline__ void mma_tile(const uint32_t (&lhs)[4], const uint32_t (&rhs)[2],
                                         float (&accum)[4]) {{
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{{%0,%1,%2,%3}}, {{%4,%5,%6,%7}}, {{%8,%9}}, {{%0,%1,%2,%3}};"
        : "+f"(accum[0]), "+f"(accum[1]), "+f"(accum[2]), "+f"(accum[3])
        : "r"(lhs[0]), "r"(lhs[1]), "r"(lhs[2]), "r"(lhs[3]), "r"(rhs[0]), "r"(rhs[1]));
}}
__global__ void short_attention(const in0_type* query, const in1_type* key,
                                const in2_type* value, out0_type* output{bias_args}) {{
    constexpr int QUERIES = {queries}, HEADS = {heads}, KEYS = {keys}, WIDTH = {width};
    constexpr int DEPTH_PAD = {padded_depth}, KEY_PAD = {padded_keys}, WIDTH_PAD = {padded_width};
    const int prefix = blockIdx.y;
    const int first = blockIdx.x * 16;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int row_group = lane >> 2, pair_col = (lane & 3) * 2;
    __shared__ __align__(32) __half q_tile[16 * DEPTH_PAD];
    __shared__ __align__(32) __half k_tile[KEY_PAD * DEPTH_PAD];
    __shared__ __align__(32) __half v_tile[KEY_PAD * WIDTH_PAD];
    __shared__ __align__(32) __half p_tile[16 * KEY_PAD];
    __shared__ float maxima[16][4], sums[16][4];
    {query_load}
    {key_load}
    {value_load}
    asm volatile("cp.async.commit_group; cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
    float scores[2][4] = {{}};
    #pragma unroll
    for (int inner = 0; inner < DEPTH_PAD; inner += 16) {{
        uint32_t packed_q[4];
        load_a(q_tile + swizzle(lane & 15, inner + (lane >> 4) * 8, DEPTH_PAD), packed_q);
        #pragma unroll
        for (int tile = 0; tile < 2; ++tile) {{
            uint32_t packed_k[2];
            load_b(k_tile + swizzle(warp * 16 + tile * 8 + (lane & 7),
                   inner + ((lane >> 3) & 1) * 8, DEPTH_PAD), packed_k);
            mma_tile(packed_q, packed_k, scores[tile]);
        }}
    }}
    float local_max[2] = {{-INFINITY, -INFINITY}};
    #pragma unroll
    for (int tile = 0; tile < 2; ++tile) {{
        #pragma unroll
        for (int tag = 0; tag < 4; ++tag) {{
            const int row = first + row_group + (tag >> 1) * 8;
            const int col = warp * 16 + tile * 8 + pair_col + (tag & 1);
            float updated = -INFINITY;
            if (row < QUERIES && col < KEYS) {{
                __half score = __float2half_rn(scores[tile][tag]);
                {bias_loads}
                updated = __half2float(score);
            }}
            scores[tile][tag] = updated;
            local_max[tag >> 1] = fmaxf(local_max[tag >> 1], updated);
        }}
    }}
    #pragma unroll
    for (int half = 0; half < 2; ++half) {{
        float high = local_max[half];
        high = fmaxf(high, __shfl_xor_sync(0xffffffff, high, 1));
        high = fmaxf(high, __shfl_xor_sync(0xffffffff, high, 2));
        if (!pair_col) maxima[row_group + half * 8][warp] = high;
    }}
    __syncthreads();
    #pragma unroll
    for (int half = 0; half < 2; ++half) {{
        const float* highs = maxima[row_group + half * 8];
        local_max[half] = fmaxf(fmaxf(highs[0], highs[1]), fmaxf(highs[2], highs[3]));
    }}
    float local_sum[2] = {{}};
    #pragma unroll
    for (int tile = 0; tile < 2; ++tile) {{
        #pragma unroll
        for (int tag = 0; tag < 4; ++tag) {{
            const int row = first + row_group + (tag >> 1) * 8;
            const int col = warp * 16 + tile * 8 + pair_col + (tag & 1);
            float exponential = row < QUERIES && col < KEYS ?
                expf(scores[tile][tag] - local_max[tag >> 1]) : 0.0f;
            scores[tile][tag] = exponential;
            local_sum[tag >> 1] += exponential;
        }}
    }}
    #pragma unroll
    for (int half = 0; half < 2; ++half) {{
        float total = local_sum[half];
        total += __shfl_xor_sync(0xffffffff, total, 1);
        total += __shfl_xor_sync(0xffffffff, total, 2);
        if (!pair_col) sums[row_group + half * 8][warp] = total;
    }}
    __syncthreads();
    #pragma unroll
    for (int half = 0; half < 2; ++half) {{
        const float* totals = sums[row_group + half * 8];
        local_sum[half] = 1.0f / ((totals[0] + totals[1]) + (totals[2] + totals[3]));
    }}
    #pragma unroll
    for (int tile = 0; tile < 2; ++tile) {{
        #pragma unroll
        for (int tag = 0; tag < 4; ++tag) {{
            const int row = row_group + (tag >> 1) * 8;
            const int col = warp * 16 + tile * 8 + pair_col + (tag & 1);
            p_tile[swizzle(row, col, KEY_PAD)] = first + row < QUERIES && col < KEYS ?
                __float2half_rn(scores[tile][tag] * local_sum[tag >> 1]) : __float2half_rn(0.0f);
        }}
    }}
    __syncthreads();
    #pragma unroll
    for (int tile = 0; tile < (WIDTH + 31) / 32; ++tile) {{
        const int out_group = tile * 32 + warp * 8;
        if (out_group >= WIDTH) continue;
        float out_pack[4] = {{}};
        #pragma unroll
        for (int inner = 0; inner < KEYS; inner += 16) {{
            uint32_t packed_p[4], packed_v[2];
            load_a(p_tile + swizzle(lane & 15, inner + (lane >> 4) * 8, KEY_PAD), packed_p);
            load_b_trans(v_tile + swizzle(inner + (lane & 15), out_group, WIDTH_PAD), packed_v);
            mma_tile(packed_p, packed_v, out_pack);
        }}
        #pragma unroll
        for (int tag = 0; tag < 4; ++tag) {{
            const int row = first + row_group + (tag >> 1) * 8;
            const int col = out_group + pair_col + (tag & 1);
            if (row < QUERIES && col < WIDTH)
                output[((prefix / HEADS * QUERIES + row) * HEADS + prefix % HEADS) * WIDTH + col]
                    = __float2half_rn(out_pack[tag]);
        }}
    }}
}}
short_attention<<<dim3({(queries+15)//16}, {batch*heads}), 128>>>(in0_p, in1_p, in2_p, out0_p{launch_args});
CHECK(0 == cudaGetLastError());
'''


def _fp16_cuda(tensor):
    return tensor.placement_backend == 1 and dtype_name(tensor.dtype) == "float16"


def _capture(tensor):
    from jittor._runtime import step_capture

    capture = step_capture._ACTIVE
    if capture is not None and jt.flags.no_grad and _fp16_cuda(tensor):
        architectures = jt.flags.cuda_archs
        if architectures and min(architectures) >= 80:
            return capture
    return None


def _stamp(tensor):
    return tensor, tensor.var_ptr, tensor._saved_storage_version()


def _unchanged(stamp):
    tensor, pointer, version = stamp
    return tensor.var_ptr == pointer and tensor._saved_storage_version() == version


def record_matmul(result, lhs, rhs):
    capture = _capture(result)
    if (capture is None or result._producer_name() != "cublas_batched_matmul"
            or not _fp16_cuda(lhs) or not _fp16_cuda(rhs)
            or lhs.shape[:2] != rhs.shape[:2] or lhs.shape[0] * lhs.shape[1] > 65535
            or any(not 16 <= dim <= 64 for dim in (*lhs.shape[-2:], rhs.shape[-1]))):
        return
    capture.short_attention[result.var_ptr] = ("matmul", _stamp(result), (_stamp(lhs), _stamp(rhs)))


def record_softmax(result, scores):
    capture = _capture(result)
    if capture is None or not capture.short_attention:
        return
    capture.short_attention[result.var_ptr] = ("softmax", _stamp(result), (_stamp(scores),))


def _source(capture, value, op):
    entry = capture.short_attention.get(value.var_ptr)
    if (entry is None or entry[0] != op or value.is_finished
            or not _unchanged(entry[1]) or not all(_unchanged(stamp) for stamp in entry[2])
            or len(value._inputs()) != len(entry[2])):
        return None
    return tuple(stamp[0] for stamp in entry[2])


def _score_operands(capture, scores):
    additions = []
    for _ in range(10):
        pair = _source(capture, scores, "matmul")
        if pair is not None:
            return pair, tuple(reversed(additions))
        op = scores._producer_op()
        if op in ("clone", "reshape") and scores._storage_is_contiguous():
            inputs = scores._inputs()
            if len(inputs) != 1:
                return None
            parent = inputs[0]
            if not parent._storage_is_contiguous() or parent.numel() != scores.numel():
                return None
            scores = parent
        elif op == "binary.add" and len(additions) < 2:
            inputs = scores._inputs()
            if len(inputs) != 2:
                return None
            lhs, rhs = inputs
            if (lhs.shape != rhs.shape or rhs.shape != scores.shape
                    or not _fp16_cuda(lhs) or not _fp16_cuda(rhs)):
                return None
            additions.append(rhs)
            scores = lhs
        else:
            return None
    return None


def _dense_root(value):
    strides = tuple(value._storage_strides())
    if not value.numel() or any(stride < 0 for stride in strides):
        return None
    root = value
    for _ in range(12):
        if root._storage_is_contiguous():
            break
        if root.is_finished or not root._producer_is_view():
            return None
        inputs = root._inputs()
        if len(inputs) != 1:
            return None
        root = inputs[0]
    else:
        return None
    shape = tuple(value.shape)
    extent = sum((length - 1) * stride for length, stride in zip(shape, strides))
    offset = value._storage_offset() - root._storage_offset()
    if (root.dtype != value.dtype or root.placement_backend != value.placement_backend
            or root.device_id != value.device_id or offset < 0 or offset + extent >= root.numel()):
        return None
    return root, (shape, strides, offset)


def try_contiguous(value):
    """Read the softmax product's sequence-major copy directly from its operands."""
    capture = _capture(value)
    if capture is None or not capture.short_attention or value.ndim != 4:
        return None
    product = value
    for _ in range(4):
        if product.is_finished or not product._producer_is_view():
            return None
        inputs = product._inputs()
        if len(inputs) != 1:
            return None
        product = inputs[0]
        operands = _source(capture, product, "matmul")
        if operands is not None:
            break
    else:
        return None
    if product.ndim != 4:
        return None
    permutation = (0, 2, 1, 3)
    product_strides = product._storage_strides()
    if (tuple(value.shape) != tuple(product.shape[axis] for axis in permutation)
            or tuple(value._storage_strides()) != tuple(product_strides[axis] for axis in permutation)
            or value._storage_offset() != product._storage_offset()):
        return None
    probabilities, values = operands
    softmax_inputs = _source(capture, probabilities, "softmax")
    if softmax_inputs is None:
        return None
    matched_scores = _score_operands(capture, softmax_inputs[0])
    if matched_scores is None:
        return None
    (query, key), additions = matched_scores
    if (query.shape[:2] != values.shape[:2] or key.shape[:2] != values.shape[:2]
            or query.shape[-2] != probabilities.shape[-2] or key.shape[-1] != probabilities.shape[-1]
            or query.shape[-1] != key.shape[-2] or values.shape[-2] != probabilities.shape[-1]
            or any(addition.shape[-2:] != probabilities.shape[-2:]
                   or addition.numel() != probabilities.numel() for addition in additions)):
        return None
    key_rows = key._storage_permute(jt.NanoVector((0, 1, 3, 2)))
    descriptors = [_dense_root(tensor) for tensor in (query, key_rows, values, *additions)]
    if any(descriptor is None for descriptor in descriptors):
        return None
    roots, layouts = zip(*descriptors)
    cuda_src = _attention_source(layouts)
    return jt.code(value.shape, value.dtype, roots, cuda_src=cuda_src,
                   cuda_header='#include <cuda_fp16.h>\n#include <type/fp16_compute.h>')
