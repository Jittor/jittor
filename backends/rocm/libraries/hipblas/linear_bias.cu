#include "linear_bias.h"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <algorithm>

namespace jittor {
namespace {
template<class T>
__global__ void linear_bias_cast(const float* product, const T* bias,
                                 T* output, int64_t count, int width) {
    for (int64_t i = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
            i < count; i += int64_t(gridDim.x)*blockDim.x)
        output[i] = T(product[i] + float(bias[i % width]));
}
}

hipError_t hip_linear_bias_cast(const float* product, const void* bias,
    void* output, int64_t count, int width, bool bf16, hipStream_t stream) {
    int blocks = int(std::min<int64_t>((count + 255) / 256, 65536));
    if (!blocks) return hipSuccess;
    if (bf16)
        linear_bias_cast<<<blocks, 256, 0, stream>>>(product,
            static_cast<const hip_bfloat16*>(bias), static_cast<hip_bfloat16*>(output), count, width);
    else
        linear_bias_cast<<<blocks, 256, 0, stream>>>(product,
            static_cast<const __half*>(bias), static_cast<__half*>(output), count, width);
    return hipGetLastError();
}
}
