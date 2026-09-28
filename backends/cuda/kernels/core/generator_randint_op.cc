#ifdef JIT_cuda
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include "core/var.h"
#include "ops/composite/generator_randint_op.h"
#include "runtime/cuda_streams.h"
#include "runtime/device.h"
#include "stream_compat.h"
#include "utils/philox.h"

namespace jittor {

__device__ static inline uint64 generator_randint_sample(uint64 seed,
        uint64 element, uint64 span, uint64 threshold) {
    const int pair = int(element & 1);
    const uint64 block_index = element >> 1;
    uint64 attempt = 0;
    while (true) {
        auto block = philox4x32_10(seed, block_index, attempt++);
        auto product = philox_mul_wide(philox_uint64(block, pair), span);
        if (product.low >= threshold) return product.high;
    }
}

template<class T>
__global__ static void generator_randint_kernel(T* output, index_t num,
        int64 low, int64 high, int64 seed, int64 offset) {
    int64 i = int64(blockIdx.x) * int64(blockDim.x) + int64(threadIdx.x);
    int64 stride = int64(blockDim.x) * int64(gridDim.x);
    const uint64 span = uint64(high) - uint64(low);
    const uint64 threshold = (uint64(0) - span) % span;
    for (; i < num; i += stride) {
        output[i] = T(uint64(low) + generator_randint_sample(uint64(seed),
            uint64(offset) + uint64(i), span, threshold));
    }
}

void GeneratorRandintOp::jit_run() {
    if (output->num == 0) return;
    int block = 256;
    int grid = (output->num + block - 1) / block;
    if (grid > 65535) grid = 65535;
    generator_randint_kernel<T><<<grid, block, 0,
        cuda_compute_stream(current_device())>>>(output->ptr<T>(),
        output->num, low, high, seed, offset);
    checkCudaErrors(cudaGetLastError());
}

} // namespace jittor
#endif
