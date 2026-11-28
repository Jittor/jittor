#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>

namespace jt_split_kv {

struct Layout {
    int64_t batch, head, row, column;

    __device__ int64_t offset(int b, int h, int r, int c) const {
        return b * batch + h * head + r * row + c * column;
    }
};

template<int Pitch>
__device__ __forceinline__ int swizzle(int row, int column) {
    constexpr int shift = Pitch >= 64 ? 0 : (Pitch == 32 ? 1 : 2);
    return row * Pitch + (column ^ (((row >> shift) & (Pitch / 8 - 1)) * 8));
}

template<class T, int Rows, int Pitch, int Width>
__device__ __forceinline__ void load_tile(
        T* tile, const T* source, Layout layout, int batch, int head,
        int first_row, int end_row) {
    if (!((layout.batch | layout.head | layout.row) & 7)
            && !(reinterpret_cast<uintptr_t>(source) & 15)) {
        for (int i = threadIdx.x * 8; i < Rows * Pitch; i += 128 * 8) {
            const int row = i / Pitch, column = i % Pitch;
            const bool valid = first_row + row < end_row && column < Width;
            const uint32_t dst = __cvta_generic_to_shared(tile + swizzle<Pitch>(row, column));
            const T* src = valid ? source + layout.offset(batch, head, first_row + row, column) : source;
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;" ::
                "r"(dst), "l"(src), "r"(valid ? 16 : 0));
        }
    } else {
        for (int i = threadIdx.x; i < Rows * Pitch; i += 128) {
            const int row = i / Pitch, column = i % Pitch;
            tile[swizzle<Pitch>(row, column)] = first_row + row < end_row && column < Width
                ? source[layout.offset(batch, head, first_row + row, column)] : T(0.0f);
        }
    }
}

template<class T>
__device__ __forceinline__ void load_a(const T* address, uint32_t (&regs)[4]) {
    const uint32_t shared = __cvta_generic_to_shared(address);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
        : "=r"(regs[0]), "=r"(regs[1]), "=r"(regs[2]), "=r"(regs[3]) : "r"(shared));
}

template<bool Transpose, class T>
__device__ __forceinline__ void load_b(const T* address, uint32_t (&regs)[2]) {
    const uint32_t shared = __cvta_generic_to_shared(address);
    if (Transpose) {
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
            : "=r"(regs[0]), "=r"(regs[1]) : "r"(shared));
    } else {
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
            : "=r"(regs[0]), "=r"(regs[1]) : "r"(shared));
    }
}

template<bool BF16>
__device__ __forceinline__ void mma(
        const uint32_t (&lhs)[4], const uint32_t (&rhs)[2], float (&accum)[4]) {
    if (BF16) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+f"(accum[0]), "+f"(accum[1]), "+f"(accum[2]), "+f"(accum[3])
            : "r"(lhs[0]), "r"(lhs[1]), "r"(lhs[2]), "r"(lhs[3]), "r"(rhs[0]), "r"(rhs[1]));
    } else {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+f"(accum[0]), "+f"(accum[1]), "+f"(accum[2]), "+f"(accum[3])
            : "r"(lhs[0]), "r"(lhs[1]), "r"(lhs[2]), "r"(lhs[3]), "r"(rhs[0]), "r"(rhs[1]));
    }
}

template<class T, class Mask, bool BooleanMask, bool BF16, int HeadDim, int ValueDim,
         int DepthPitch, int ValuePitch, int TileKeys>
__global__ __launch_bounds__(128) void partial_attention(
        const T* query, const T* key, const T* value, const Mask* mask,
        Layout q_layout, Layout k_layout, Layout v_layout, Layout mask_layout,
        int queries, int keys, int heads, int group, float scale,
        float* output, float2* statistics) {
    constexpr int ScoresPerWarp = TileKeys / (4 * 8);
    const int prefix = blockIdx.y, split = blockIdx.z;
    const int splits = gridDim.z;
    const int batch = prefix / heads, head = prefix % heads, kv_head = head / group;
    const int first_query = blockIdx.x * 16, first_key = split * TileKeys;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int row_group = lane >> 2, pair_column = (lane & 3) * 2;
    __shared__ __align__(16) T q_tile[16 * DepthPitch];
    __shared__ __align__(16) T k_tile[TileKeys * DepthPitch];
    __shared__ __align__(16) T v_tile[TileKeys * ValuePitch];
    __shared__ __align__(16) T p_tile[16 * TileKeys];
    __shared__ float maxima[16][4], sums[16][4];

    load_tile<T, 16, DepthPitch, HeadDim>(
        q_tile, query, q_layout, batch, head, first_query, queries);
    load_tile<T, TileKeys, DepthPitch, HeadDim>(
        k_tile, key, k_layout, batch, kv_head, first_key, keys);
    asm volatile("cp.async.commit_group;" ::: "memory");
    load_tile<T, TileKeys, ValuePitch, ValueDim>(
        v_tile, value, v_layout, batch, kv_head, first_key, keys);
    asm volatile("cp.async.commit_group; cp.async.wait_group 0;" ::: "memory");
    __syncthreads();

    float scores[ScoresPerWarp][4] = {};
    #pragma unroll
    for (int inner = 0; inner < DepthPitch; inner += 16) {
        uint32_t q[4];
        load_a(q_tile + swizzle<DepthPitch>(lane & 15, inner + (lane >> 4) * 8), q);
        #pragma unroll
        for (int tile = 0; tile < ScoresPerWarp; ++tile) {
            uint32_t k[2];
            const int row = (warp * ScoresPerWarp + tile) * 8 + (lane & 7);
            load_b<false>(k_tile + swizzle<DepthPitch>(row, inner + ((lane >> 3) & 1) * 8), k);
            mma<BF16>(q, k, scores[tile]);
        }
    }

    float high[2] = {-CUDART_INF_F, -CUDART_INF_F};
    #pragma unroll
    for (int tile = 0; tile < ScoresPerWarp; ++tile) {
        #pragma unroll
        for (int item = 0; item < 4; ++item) {
            const int row = first_query + row_group + (item >> 1) * 8;
            const int column = first_key + (warp * ScoresPerWarp + tile) * 8
                + pair_column + (item & 1);
            float score = -CUDART_INF_F;
            if (row < queries && column < keys) {
                score = scores[tile][item] * scale;
                if (mask) {
                    const auto bias = mask[mask_layout.offset(batch, head, row, column)];
                    if (BooleanMask) {
                        if (static_cast<float>(bias) == 0.0f) score = -CUDART_INF_F;
                    } else {
                        score += static_cast<float>(bias);
                    }
                }
            }
            scores[tile][item] = score;
            high[item >> 1] = fmaxf(high[item >> 1], score);
        }
    }
    #pragma unroll
    for (int half = 0; half < 2; ++half) {
        float top = fmaxf(high[half], __shfl_xor_sync(0xffffffff, high[half], 1));
        top = fmaxf(top, __shfl_xor_sync(0xffffffff, top, 2));
        if (!pair_column) maxima[row_group + half * 8][warp] = top;
    }
    __syncthreads();
    #pragma unroll
    for (int half = 0; half < 2; ++half) {
        const float* tops = maxima[row_group + half * 8];
        high[half] = fmaxf(fmaxf(tops[0], tops[1]), fmaxf(tops[2], tops[3]));
    }

    float sum[2] = {};
    #pragma unroll
    for (int tile = 0; tile < ScoresPerWarp; ++tile) {
        #pragma unroll
        for (int item = 0; item < 4; ++item) {
            const float score = scores[tile][item];
            const float weight = score == -CUDART_INF_F ? 0.0f : expf(score - high[item >> 1]);
            const int row = row_group + (item >> 1) * 8;
            const int column = (warp * ScoresPerWarp + tile) * 8 + pair_column + (item & 1);
            p_tile[swizzle<TileKeys>(row, column)] = T(weight);
            sum[item >> 1] += weight;
        }
    }
    #pragma unroll
    for (int half = 0; half < 2; ++half) {
        float total = sum[half] + __shfl_xor_sync(0xffffffff, sum[half], 1);
        total += __shfl_xor_sync(0xffffffff, total, 2);
        if (!pair_column) sums[row_group + half * 8][warp] = total;
    }
    __syncthreads();

    if (warp == 0 && !pair_column) {
        #pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int row = row_group + half * 8;
            if (first_query + row < queries) {
                const float* totals = sums[row];
                const int index = ((prefix * queries + first_query + row) * splits) + split;
                statistics[index] = make_float2(high[half],
                    (totals[0] + totals[1]) + (totals[2] + totals[3]));
            }
        }
    }
    #pragma unroll
    for (int tile = 0; tile < (ValueDim + 31) / 32; ++tile) {
        const int column_group = tile * 32 + warp * 8;
        if (column_group >= ValueDim) continue;
        float result[4] = {};
        #pragma unroll
        for (int inner = 0; inner < TileKeys; inner += 16) {
            uint32_t p[4], v[2];
            load_a(p_tile + swizzle<TileKeys>(lane & 15, inner + (lane >> 4) * 8), p);
            load_b<true>(v_tile + swizzle<ValuePitch>(inner + (lane & 15), column_group), v);
            mma<BF16>(p, v, result);
        }
        #pragma unroll
        for (int item = 0; item < 4; ++item) {
            const int row = first_query + row_group + (item >> 1) * 8;
            const int column = column_group + pair_column + (item & 1);
            if (row < queries && column < ValueDim) {
                const int index = (prefix * queries + row) * splits + split;
                output[index * ValueDim + column] = result[item];
            }
        }
    }
}

template<class T, int Width, bool SequenceMajor>
__global__ void combine_attention(
        const float* partials, const float2* statistics, T* output,
        int queries, int heads, int splits) {
    const int row = blockIdx.x;
    __shared__ float weights[64];
    __shared__ float total;
    const float2* stats = statistics + row * splits;
    if (!threadIdx.x) {
        float top = -CUDART_INF_F;
        for (int split = 0; split < splits; ++split)
            top = fmaxf(top, stats[split].x);
        float sum = 0.0f;
        for (int split = 0; split < splits; ++split) {
            const float2 stat = stats[split];
            const float weight = stat.y == 0.0f ? 0.0f : expf(stat.x - top);
            weights[split] = weight;
            sum += weight * stat.y;
        }
        total = sum;
    }
    __syncthreads();
    int output_row = row;
    if (SequenceMajor) {
        const int batch = row / (queries * heads);
        const int head = row / queries % heads;
        output_row = (batch * queries + row % queries) * heads + head;
    }
    const int column = threadIdx.x;
    if (column < Width) {
        float result = 0.0f;
        if (total != 0.0f) {
            const float* partial = partials + row * splits * Width + column;
            for (int split = 0; split < splits; ++split) {
                if (weights[split] != 0.0f)
                    result += weights[split] * partial[split * Width];
            }
            result /= total;
        }
        output[output_row * Width + column] = T(result);
    }
}

} // namespace jt_split_kv
