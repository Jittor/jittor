#include "convolution.h"
#include <hip/hip_runtime.h>
#include <algorithm>

namespace jittor {
namespace {
__global__ void columns_kernel(const float* x, float* columns, int64_t count,
    int channels, int height, int width, int kh, int kw, int oh, int ow,
    int sh, int sw, int ph, int pw, int dh, int dw) {
    const int64_t spatial = (int64_t)oh * ow, inner = (int64_t)channels * kh * kw;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
            i < count; i += (int64_t)gridDim.x * blockDim.x) {
        const int64_t p = i % spatial, k = i / spatial % inner, n = i / spatial / inner;
        const int64_t h = p / ow * sh - ph + k / kw % kh * dh;
        const int64_t w = p % ow * sw - pw + k % kw * dw;
        columns[i] = h >= 0 && h < height && w >= 0 && w < width
            ? x[((n * channels + k / (kh * kw)) * height + h) * width + w] : 0.f;
    }
}

__global__ void bias_kernel(float* y, const float* bias, int64_t count,
                          int channels, int spatial) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
            i < count; i += (int64_t)gridDim.x * blockDim.x)
        y[i] += bias[i / spatial % channels];
}

unsigned blocks(int64_t count) { return (unsigned)std::min<int64_t>((count+255)/256, 65535); }
} // namespace

hipError_t hip_conv_columns(const float* x, float* columns, int64_t count,
    int channels, int height, int width, int kh, int kw, int oh, int ow,
    int sh, int sw, int ph, int pw, int dh, int dw, hipStream_t stream) {
    columns_kernel<<<blocks(count), 256, 0, stream>>>(x, columns, count,
        channels, height, width, kh, kw, oh, ow, sh, sw, ph, pw, dh, dw);
    return hipGetLastError();
}

hipError_t hip_conv_bias(float* y, const float* bias, int64_t count,
                        int channels, int spatial, hipStream_t stream) {
    bias_kernel<<<blocks(count), 256, 0, stream>>>(y, bias, count, channels, spatial);
    return hipGetLastError();
}
} // namespace jittor
