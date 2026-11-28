"""FP32 CUDA batched matrix multiplication for narrow outputs."""

from math import prod

import jittor as jt
from jittor._runtime.dispatch import optional_kernel


def _supports_narrow_batched_matmul(a, b):
    if (a.ndim < 3 or a.ndim != b.ndim or a.shape[:-2] != b.shape[:-2]
            or a.shape[-1] != b.shape[-2] or a.dtype != b.dtype):
        return False
    batch = prod(a.shape[:-2])
    rows, inner = a.shape[-2:]
    columns = b.shape[-1]
    # A wide vendor tile leaves too few blocks when both output axes and
    # the batch are small. A row tile of eight exposes more parallel work.
    return (2 <= batch <= 32 and 16 <= rows <= 128 and 16 <= columns <= 32
            and 64 <= inner <= 2048
            and a._storage_is_contiguous() and b._storage_is_contiguous()
            and not (jt.flags.amp_reg & ~jt.amp_flags.keep_reduce)
            and jt.core.float32_precision_state()[0] == "highest")


_SOURCE = r"""
constexpr int tile_rows = 8;
constexpr int tile_columns = 32;
constexpr int tile_inner = 64;

__global__ static void narrow_batched_product(const float* a, const float* b, float* c,
                                              int rows, int columns, int inner) {
    __shared__ float left[tile_rows][tile_inner];
    __shared__ float right[tile_inner][tile_columns];
    const int x = threadIdx.x, y = threadIdx.y;
    const int row = blockIdx.y * tile_rows + y;
    const int column = x;
    a += int64_t(blockIdx.z) * rows * inner;
    b += int64_t(blockIdx.z) * inner * columns;
    c += int64_t(blockIdx.z) * rows * columns;
    float value = 0.0f;
    for (int start = 0; start < inner; start += tile_inner) {
        for (int k = x; k < tile_inner; k += tile_columns)
            left[y][k] = row < rows && start + k < inner ? a[row * inner + start + k] : 0.0f;
        for (int k = y; k < tile_inner; k += tile_rows)
            right[k][x] = column < columns && start + k < inner
                ? b[(start + k) * columns + column] : 0.0f;
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < tile_inner; ++k)
            value = __fmaf_rn(left[y][k], right[k][x], value);
        __syncthreads();
    }
    if (row < rows && column < columns) c[row * columns + column] = value;
}
"""


@optional_kernel("nn.narrow_batched_matmul", "cuda", dtypes=("float32",),
                 supports=_supports_narrow_batched_matmul)
def narrow_batched_matmul(a, b):
    rows, inner = a.shape[-2:]
    columns = b.shape[-1]
    source = f"""
    const dim3 threads(tile_columns, tile_rows);
    const dim3 blocks(1, ({rows} + tile_rows - 1) / tile_rows, {prod(a.shape[:-2])});
    narrow_batched_product<<<blocks, threads>>>(in0_p, in1_p, out0_p, {rows}, {columns}, {inner});
    CHECK(0 == cudaGetLastError());
    """
    return jt.code((*a.shape[:-1], columns), a.dtype, [a, b],
                   cuda_header=_SOURCE, cuda_src=source, backend="cuda")
