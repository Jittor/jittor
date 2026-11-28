#pragma once
#include <hip/hip_runtime_api.h>
#include <cstdint>

namespace jittor {
hipError_t hip_linear_bias_cast(const float* product, const void* bias,
    void* output, int64_t count, int width, bool bf16, hipStream_t stream);
}
