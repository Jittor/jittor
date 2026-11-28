"""FP32 CUDA batched matrix multiplication for small output tiles."""

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


def _supports_short_batched_matmul(a, b, transposed_right):
    if (a.ndim < 3 or a.ndim != b.ndim or a.shape[:-2] != b.shape[:-2]
            or a.dtype != b.dtype
            or jt.flags.amp_reg & ~jt.amp_flags.keep_reduce
            or jt.core.float32_precision_state()[0] != "highest"):
        return False
    rows, inner = a.shape[-2:]
    columns, right_inner = b.shape[-2:] if transposed_right else b.shape[-2:][::-1]
    return (inner == right_inner and 16 <= inner <= 64
            and 16 <= rows <= 64 and 16 <= columns <= 64
            and 32 <= prod(a.shape[:-2]) <= 65535)


_SHORT_PRODUCT = r"""
struct BatchMatrix {
    int batches_per_group;
    int64_t outer_stride, inner_stride, row_stride, offset;

    __device__ int64_t batch_offset(int batch) const {
        return offset + (batch / batches_per_group) * outer_stride
            + (batch % batches_per_group) * inner_stride;
    }
};

template<int Rows, int Columns, int Inner, bool TransposedRight>
__global__ static void short_batched_product(const float* a, const float* b, float* c,
                                             BatchMatrix left, BatchMatrix right) {
    constexpr int TileRows = 16;
    constexpr int TileCols = 32;
    constexpr int TileInner = Inner <= 32 ? 32 : 64;
    __shared__ float lhs[TileRows][TileInner];
    __shared__ float rhs[TileInner][TileCols + 1];

    a += left.batch_offset(blockIdx.z);
    b += right.batch_offset(blockIdx.z);
    c += (size_t)blockIdx.z * Rows * Columns;
    const int first_row = blockIdx.y * TileRows;
    const int first_col = blockIdx.x * TileCols;
    for (int item = threadIdx.x; item < TileRows * TileInner; item += 256) {
        const int r = item / TileInner, k = item % TileInner;
        lhs[r][k] = first_row + r < Rows && k < Inner
            ? a[(first_row + r) * left.row_stride + k] : 0.0f;
    }
    for (int item = threadIdx.x; item < TileInner * TileCols; item += 256) {
        const int k = TransposedRight ? item % TileInner : item / TileCols;
        const int col = TransposedRight ? item / TileInner : item % TileCols;
        const int j = first_col + col;
        rhs[k][col] = k < Inner && j < Columns
            ? b[TransposedRight ? j * right.row_stride + k : k * right.row_stride + j] : 0.0f;
    }
    __syncthreads();
    const int row = threadIdx.x / TileCols;
    const int col = threadIdx.x % TileCols;
    if (first_col + col >= Columns) return;
    float upper = 0.0f, lower = 0.0f;
    #pragma unroll
    for (int k = 0; k < Inner; ++k) {
        const float value = rhs[k][col];
        upper = __fmaf_rn(lhs[row][k], value, upper);
        lower = __fmaf_rn(lhs[row + 8][k], value, lower);
    }
    if (first_row + row < Rows)
        c[(first_row + row) * Columns + first_col + col] = upper;
    if (first_row + row + 8 < Rows)
        c[(first_row + row + 8) * Columns + first_col + col] = lower;
}
"""


def _batch_matrix(tensor):
    if tensor._storage_is_contiguous():
        rows, cols = tensor.shape[-2:]
        return tensor, (1, rows * cols, 0, cols, 0)
    if tensor.ndim > 4:
        return None
    from .fused_attention_f32_cuda import _dense_storage_layout

    layout = _dense_storage_layout(tensor)
    if layout is None:
        return None
    base, strides, offset = layout
    outer = strides[0] if tensor.ndim == 4 else 0
    return base, (tensor.shape[-3], outer, strides[-3], strides[-2], offset)


@optional_kernel("nn.short_batched_matmul", "cuda", dtypes=("float32",),
                 supports=_supports_short_batched_matmul)
def short_batched_matmul(a, b, transposed_right):
    left, right = _batch_matrix(a), _batch_matrix(b)
    if left is None or right is None:
        return None
    left_root, left_layout = left
    right_root, right_layout = right
    rows, inner = a.shape[-2:]
    columns = b.shape[-2] if transposed_right else b.shape[-1]
    transposed = "true" if transposed_right else "false"
    source = f"""
    const dim3 blocks(({columns} + 31) / 32, ({rows} + 15) / 16, {prod(a.shape[:-2])});
    short_batched_product<{rows}, {columns}, {inner}, {transposed}><<<blocks, 256>>>(
        in0_p, in1_p, out0_p,
        BatchMatrix{{{', '.join(map(str, left_layout))}}},
        BatchMatrix{{{', '.join(map(str, right_layout))}}});
    CHECK(0 == cudaGetLastError());
    """
    return jt.code((*a.shape[:-1], columns), a.dtype, [left_root, right_root],
                   cuda_header=_SHORT_PRODUCT, cuda_src=source, backend="cuda")
