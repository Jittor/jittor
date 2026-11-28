"""Fused first-dimension FP32 weight normalization on CUDA."""

import jittor as jt
from jittor._runtime.core_api import _output_requires_grad
from jittor._runtime.dispatch import optional_kernel


_HEADER = """
__global__ void weight_norm_rows(const float* direction, const float* gain,
                                 float* weight, int64_t width) {
    constexpr int threads = 256;
    int lane = threadIdx.x;
    int64_t base = static_cast<int64_t>(blockIdx.x) * width;
    float sum = 0.f;
    for (int64_t column = lane; column < width; column += threads) {
        float value = direction[base + column];
        sum = __fmaf_rn(value, value, sum);
    }

    __shared__ float partial[threads];
    partial[lane] = sum;
    __syncthreads();
    for (int stride = threads / 2; stride >= 32; stride >>= 1) {
        if (lane < stride) partial[lane] += partial[lane + stride];
        __syncthreads();
    }
    if (lane < 32) {
        sum = partial[lane];
        for (int stride = 16; stride; stride >>= 1)
            sum += __shfl_down_sync(0xffffffffu, sum, stride);
        if (lane == 0) partial[0] = __fdiv_rn(1.f, __fsqrt_rn(sum));
    }
    __syncthreads();

    float scale = gain[blockIdx.x];
    float inv_norm = partial[0];
    for (int64_t column = lane; column < width; column += threads)
        weight[base + column] = (scale * direction[base + column]) * inv_norm;
}
"""


def _supports_weight_norm_rows(direction, gain):
    return (
        not jt.flags.amp_reg
        and not _output_requires_grad(direction, gain)
        and direction.ndim > 1
        and 0 < direction.shape[0] <= 2147483647
        and direction.numel() > 0
        and tuple(gain.shape) == (direction.shape[0],) + (1,) * (direction.ndim - 1)
        and direction._storage_is_contiguous()
        and gain._storage_is_contiguous()
    )


@optional_kernel("weight_norm_rows", "cuda", dtypes={"float32"},
                 supports=_supports_weight_norm_rows)
def weight_norm_rows(direction, gain):
    # This is Torch's fused CUDA rounding order: accumulate squares with
    # FMA, round the reciprocal of the norm, then multiply by gain and the
    # reciprocal separately. A block handles one contiguous weight row.
    return jt.code(
        direction.shape, direction.dtype, [direction, gain],
        cuda_header=_HEADER,
        cuda_src="""
            weight_norm_rows<<<in0_shape0, 256>>>(
                in0_p, in1_p, out0_p, in0->num / in0_shape0);
        """,
    )
