#pragma once
#include <cstdint>
#include <hip/hip_runtime_api.h>

namespace jittor {
hipError_t hip_conv_columns(const float* x, float* columns, int64_t count,
    int channels, int height, int width, int kh, int kw, int oh, int ow,
    int sh, int sw, int ph, int pw, int dh, int dw, hipStream_t stream);
hipError_t hip_conv_bias(float* y, const float* bias, int64_t count,
                        int channels, int spatial, hipStream_t stream);
} // namespace jittor
