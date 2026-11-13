#include "arg_reduce.h"
#include <hip/hip_runtime.h>
#include <cstdint>

namespace jittor {
namespace {
template<class T>
__device__ bool prefer(T candidate, int candidate_index, T best, int best_index, bool minimum) {
    if (candidate_index < 0) return false;
    if (best_index < 0) return true;
    const bool candidate_nan = candidate != candidate, best_nan = best != best;
    if (candidate_nan != best_nan) return candidate_nan;
    if (candidate_nan || candidate == best) return candidate_index < best_index;
    return minimum ? candidate < best : candidate > best;
}

template<class T>
__global__ void reduce_rows(const T* input, const int* offsets, int rows,
                           T* values, int* indices, bool minimum) {
    __shared__ T partial_values[256];
    __shared__ int partial_indices[256];
    const int thread = threadIdx.x;
    for (int row = blockIdx.x; row < rows; row += gridDim.x) {
        const int begin = offsets[row], end = offsets[row + 1];
        T best = T(0);
        int best_index = -1;
        for (int64_t position = int64_t(begin) + thread; position < end; position += blockDim.x) {
            const T candidate = input[position];
            const int index = int(position - begin);
            if (prefer(candidate, index, best, best_index, minimum)) {
                best = candidate;
                best_index = index;
            }
        }
        partial_values[thread] = best;
        partial_indices[thread] = best_index;
        __syncthreads();
        for (int distance = 128; distance; distance >>= 1) {
            if (thread < distance && prefer(partial_values[thread + distance],
                    partial_indices[thread + distance], partial_values[thread],
                    partial_indices[thread], minimum)) {
                partial_values[thread] = partial_values[thread + distance];
                partial_indices[thread] = partial_indices[thread + distance];
            }
            __syncthreads();
        }
        if (!thread) {
            values[row] = partial_values[0];
            indices[row] = partial_indices[0];
        }
        __syncthreads();
    }
}

template<class T>
void launch(const void* input, const int* offsets, int rows, void* values,
            int* indices, bool minimum, hipStream_t stream) {
    reduce_rows<T><<<rows < 65535 ? rows : 65535, 256, 0, stream>>>(
        static_cast<const T*>(input), offsets, rows, static_cast<T*>(values), indices, minimum);
}
} // namespace

hipError_t rocm_arg_reduce(const void* input, const int* offsets, int rows,
                          void* values, int* indices, RocprimScanType dtype,
                          bool minimum, hipStream_t stream) {
    if (!rows) return hipSuccess;
    switch (dtype) {
        case RocprimScanType::Float32: launch<float>(input, offsets, rows, values, indices, minimum, stream); break;
        case RocprimScanType::Float64: launch<double>(input, offsets, rows, values, indices, minimum, stream); break;
        case RocprimScanType::Int32: launch<int32_t>(input, offsets, rows, values, indices, minimum, stream); break;
        case RocprimScanType::Int64: launch<int64_t>(input, offsets, rows, values, indices, minimum, stream); break;
        default: return hipErrorInvalidValue;
    }
    return hipGetLastError();
}
} // namespace jittor
