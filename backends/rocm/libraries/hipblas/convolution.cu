#include "convolution.h"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <algorithm>

namespace jittor {
namespace {
template<class T>
__global__ void columns_kernel(const T* x, T* columns, int64_t count,
    int channels, int height, int width, int kh, int kw, int tile_spatial, int ow,
    int spatial_offset, int sh, int sw, int ph, int pw, int dh, int dw) {
    const int64_t inner = (int64_t)channels * kh * kw;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
            i < count; i += (int64_t)gridDim.x * blockDim.x) {
        const int64_t p = i % tile_spatial + spatial_offset;
        const int64_t k = i / tile_spatial % inner, n = i / tile_spatial / inner;
        const int64_t h = p / ow * sh - ph + k / kw % kh * dh;
        const int64_t w = p % ow * sw - pw + k % kw * dw;
        columns[i] = h >= 0 && h < height && w >= 0 && w < width
            ? x[((n * channels + k / (kh * kw)) * height + h) * width + w] : T(0.f);
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
    int channels, int height, int width, int kh, int kw, int tile_spatial, int ow,
    int spatial_offset, int sh, int sw, int ph, int pw, int dh, int dw, hipStream_t stream) {
    columns_kernel<<<blocks(count), 256, 0, stream>>>(x, columns, count,
        channels, height, width, kh, kw, tile_spatial, ow, spatial_offset, sh, sw, ph, pw, dh, dw);
    return hipGetLastError();
}

hipError_t hip_conv_columns_half(const void* x, void* columns, int64_t count,
    int channels, int height, int width, int kh, int kw, int tile_spatial, int ow,
    int spatial_offset, int sh, int sw, int ph, int pw, int dh, int dw,
    bool bf16, hipStream_t stream) {
    if (!count) return hipSuccess;
    if (bf16)
        columns_kernel<<<blocks(count),256,0,stream>>>(
            static_cast<const hip_bfloat16*>(x), static_cast<hip_bfloat16*>(columns),
            count,channels,height,width,kh,kw,tile_spatial,ow,spatial_offset,sh,sw,ph,pw,dh,dw);
    else
        columns_kernel<<<blocks(count),256,0,stream>>>(
            static_cast<const __half*>(x), static_cast<__half*>(columns),
            count,channels,height,width,kh,kw,tile_spatial,ow,spatial_offset,sh,sw,ph,pw,dh,dw);
    return hipGetLastError();
}

hipError_t hip_conv_bias(float* y, const float* bias, int64_t count,
                        int channels, int spatial, hipStream_t stream) {
    bias_kernel<<<blocks(count), 256, 0, stream>>>(y, bias, count, channels, spatial);
    return hipGetLastError();
}
} // namespace jittor
