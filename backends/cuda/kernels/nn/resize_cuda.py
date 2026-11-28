"""Separable FP32 bicubic resize for CUDA inference."""

import math

from jittor._core.flags import _output_requires_grad, flags
from jittor._runtime.dispatch import optional_kernel


_BICUBIC_HEADER = r"""
struct CubicTaps {
    int indices[4];
    float weights[4];
};

__device__ static inline float cubic_inner(float x) {
    float square = x * x;
    return 1.25f * (square * x) - 2.25f * square + 1.0f;
}

__device__ static inline float cubic_outer(float x) {
    float square = x * x;
    return -0.75f * (square * x) + 3.75f * square - 6.0f * x + 3.0f;
}

__device__ static inline CubicTaps cubic_taps(
        int output_index, int input_size, int output_size, bool align_corners) {
    const float pos = align_corners
        ? float(output_index) * (input_size - 1) / max(1, output_size - 1)
        : (float(output_index) + 0.5f) * input_size / output_size - 0.5f;
    const int start = int(floorf(pos));
    const float fraction = pos - start;
    CubicTaps taps;
    #pragma unroll
    for (int k = 0; k < 4; ++k)
        taps.indices[k] = min(max(start + k - 1, 0), input_size - 1);
    taps.weights[0] = cubic_outer(fraction + 1.0f);
    taps.weights[1] = cubic_inner(fraction);
    taps.weights[2] = cubic_inner(1.0f - fraction);
    taps.weights[3] = cubic_outer(2.0f - fraction);
    return taps;
}

__device__ static inline float cubic_sum(float a, float b, float c, float d,
                                         const CubicTaps& taps) {
    return a * taps.weights[0] + b * taps.weights[1] +
        c * taps.weights[2] + d * taps.weights[3];
}

__global__ static void bicubic_resize_width(const float* input, float* workspace,
                                            uint32_t count, uint32_t w, uint32_t W,
                                            bool align_corners) {
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += blockDim.x * gridDim.x) {
        const uint32_t row = i / W;
        CubicTaps taps = cubic_taps(i - row * W, w, W, align_corners);
        const float* src = input + row * w;
        workspace[i] = cubic_sum(src[taps.indices[0]], src[taps.indices[1]],
                                  src[taps.indices[2]], src[taps.indices[3]], taps);
    }
}

template<class Pack>
__global__ static void bicubic_resize_height(const Pack* workspace, Pack* output,
                                             uint32_t count, uint32_t h, uint32_t H,
                                             uint32_t width, bool align_corners) {
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += blockDim.x * gridDim.x) {
        const uint32_t row = i / width;
        const uint32_t plane = row / H;
        CubicTaps taps = cubic_taps(row - plane * H, h, H, align_corners);
        const Pack* src = workspace + plane * h * width + (i - row * width);
        const Pack a = src[taps.indices[0] * width];
        const Pack b = src[taps.indices[1] * width];
        const Pack c = src[taps.indices[2] * width];
        const Pack d = src[taps.indices[3] * width];
        Pack result;
        #pragma unroll
        for (int lane = 0; lane < sizeof(Pack) / sizeof(float); ++lane) {
            reinterpret_cast<float*>(&result)[lane] = cubic_sum(
                reinterpret_cast<const float*>(&a)[lane], reinterpret_cast<const float*>(&b)[lane],
                reinterpret_cast<const float*>(&c)[lane], reinterpret_cast<const float*>(&d)[lane],
                taps);
        }
        output[i] = result;
    }
}
"""

_BICUBIC_SOURCE = r"""
constexpr uint32_t threads = 256;
constexpr uint32_t max_blocks = 4096;
const uint32_t workspace_size = out1->num;
const uint32_t output_size = out0->num;
const uint32_t width = out0_shape3;
const bool align_corners = data["align_corners"];
const auto horizontal_blocks = std::min((workspace_size - 1) / threads + 1, max_blocks);
bicubic_resize_width<<<horizontal_blocks, threads>>>(
    in0_p, out1_p, workspace_size, in0_shape3, width, align_corners);
if (width % 4 == 0 && (((size_t)out0_p | (size_t)out1_p) & 15) == 0) {
    const auto blocks = std::min((output_size / 4 - 1) / threads + 1, max_blocks);
    bicubic_resize_height<<<blocks, threads>>>(
        reinterpret_cast<const float4*>(out1_p), reinterpret_cast<float4*>(out0_p),
        output_size / 4, in0_shape2, out0_shape2, width / 4, align_corners);
} else {
    const auto blocks = std::min((output_size - 1) / threads + 1, max_blocks);
    bicubic_resize_height<<<blocks, threads>>>(
        out1_p, out0_p, output_size, in0_shape2, out0_shape2, width, align_corners);
}
"""


def _supports_bicubic(img, size, mode, align_corners, tf_mode):
    if (mode != "bicubic" or flags.amp_reg or _output_requires_grad(img)
            or not img._storage_is_contiguous()):
        return False
    n, c, h, w = img.shape
    H, W = size
    # Keep the intermediate at most output-sized and the indices in 32 bits.
    return (H >= h and 65536 <= n * c * H * W < (1 << 31)
            and math.prod((n, c, h, w)) < (1 << 31))


@optional_kernel("nn.resize", "cuda", dtypes=("float32",), supports=_supports_bicubic)
def _resize_bicubic_cuda(img, size, mode, align_corners, tf_mode):
    import jittor as jt

    n, c, h, _ = img.shape
    H, W = size
    result, _ = jt.code([[n, c, H, W], [n, c, h, W]], [img.dtype, img.dtype], [img],
                        cuda_header=_BICUBIC_HEADER, cuda_src=_BICUBIC_SOURCE,
                        data={"align_corners": float(align_corners)}, backend="cuda")
    return result.stop_grad()
