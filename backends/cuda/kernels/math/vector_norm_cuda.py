"""CUDA Euclidean norms over dense runs of axes followed by columns."""

from math import prod

import jittor as jt
from jittor._runtime.core_api import _output_requires_grad
from jittor._runtime.dispatch import register_kernel


_ROWS_PER_SEGMENT = 128
_MAX_GRID_DIM = 65535


_HEADER = f"constexpr int rows_per_segment = {_ROWS_PER_SEGMENT};\n" + r"""
__device__ __forceinline__ float4 add4(float4 a, float4 b) {
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

template <bool FINAL>
__global__ void vector_norm_columns(const float* x, float4* partial,
                                    int rows, int vectors) {
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int vec = blockIdx.x * 32 + lane;
    const int begin = blockIdx.y * rows_per_segment;
    const int end = min(begin + rows_per_segment, rows);
    const size_t base = (size_t)blockIdx.z * rows * vectors + vec;
    const bool aligned = (reinterpret_cast<uintptr_t>(x) & 15u) == 0;
    float4 sum = {};
    if (vec < vectors) {
        for (int row = begin + warp; row < end; row += blockDim.y) {
            const size_t index = base + (size_t)row * vectors;
            const float* values = x + index * 4;
            float4 v = aligned ? reinterpret_cast<const float4*>(x)[index]
                               : make_float4(values[0], values[1], values[2], values[3]);
            sum.x += v.x * v.x;
            sum.y += v.y * v.y;
            sum.z += v.z * v.z;
            sum.w += v.w * v.w;
        }
    }
    __shared__ float4 sums[8][32];
    sums[warp][lane] = sum;
    __syncthreads();
    if (warp == 0 && vec < vectors) {
        #pragma unroll
        for (int i = 1; i < 8; ++i)
            sum = add4(sum, sums[i][lane]);
        if (FINAL)
            sum = make_float4(sqrtf(sum.x), sqrtf(sum.y), sqrtf(sum.z), sqrtf(sum.w));
        partial[((size_t)blockIdx.z * gridDim.y + blockIdx.y) * vectors + vec] = sum;
    }
}

__global__ void vector_norm_finish(const float4* partial, float4* out,
                                   int segments, int vectors) {
    const int vec = blockIdx.x * blockDim.x + threadIdx.x;
    if (vec >= vectors) return;
    const size_t base = (size_t)blockIdx.y * segments * vectors + vec;
    float4 sum = {};
    for (int i = 0; i < segments; ++i)
        sum = add4(sum, partial[base + (size_t)i * vectors]);
    out[(size_t)blockIdx.y * vectors + vec] =
        make_float4(sqrtf(sum.x), sqrtf(sum.y), sqrtf(sum.z), sqrtf(sum.w));
}
"""


def vector_norm_columns_cuda(x, ord, dim, keepdim):
    if (ord != 2 or x.ndim < 2 or jt.flags.amp_reg or x.numel() < (1 << 18)
            or _output_requires_grad(x) or not x._storage_is_contiguous()):
        return None
    if type(dim) is int:
        axes = (dim,)
    elif isinstance(dim, (tuple, list)):
        axes = dim
    else:
        return None
    if not axes or any(type(d) is not int or not -x.ndim <= d < x.ndim for d in axes):
        return None
    axes = sorted(d % x.ndim for d in axes)
    first, end = axes[0], axes[-1] + 1
    if axes != list(range(first, end)) or end == x.ndim:
        return None
    shape = tuple(x.shape)
    batches, rows, columns = prod(shape[:first]), prod(shape[first:end]), prod(shape[end:])
    if (not 0 < batches <= _MAX_GRID_DIM
            or not 32 <= rows <= _ROWS_PER_SEGMENT * _MAX_GRID_DIM
            or not 128 <= columns < (1 << 31) or columns % 4):
        return None
    output_shape = shape[:first] + ((1,) * len(axes) if keepdim else ()) + shape[end:]
    # Adjacent lanes read adjacent columns. Segment the reduction so that
    # even long rows leave enough blocks in flight to fill the device.
    segments = (rows + _ROWS_PER_SEGMENT - 1) // _ROWS_PER_SEGMENT
    vectors = columns // 4
    tile_launch = f"dim3(({vectors} + 31) / 32, {segments}, {batches}), dim3(32, 8)"
    if segments == 1:
        return jt.code(output_shape, x.dtype, [x], cuda_header=_HEADER, cuda_src=f"""
            vector_norm_columns<true><<<{tile_launch}>>>(
                in0_p, (float4*)out0_p, {rows}, {vectors});
        """)
    result, _ = jt.code([output_shape, (batches, segments, columns)], [x.dtype, x.dtype],
                        [x], cuda_header=_HEADER, cuda_src=f"""
        vector_norm_columns<false><<<{tile_launch}>>>(
            in0_p, (float4*)out1_p, {rows}, {vectors});
        vector_norm_finish<<<dim3(({vectors} + 255) / 256, {batches}), 256>>>(
            (const float4*)out1_p, (float4*)out0_p, {segments}, {vectors});
    """)
    return result


register_kernel("linalg.vector_norm", "cuda", vector_norm_columns_cuda, dtypes={"float32"})
