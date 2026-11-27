#include <algorithm>
#include "ops/composite/contiguous_op.h"
#include "core/var.h"

namespace jittor {
static int transpose_column_begin(const Var* input, int unit_axis, int64& columns) {
    columns = input->shape[unit_axis];
    while (unit_axis > 0) {
        const int axis = unit_axis - 1;
        if (input->shape[axis] != 1 && input->storage_stride(axis) != columns) break;
        columns *= input->shape[axis];
        unit_axis = axis;
    }
    return unit_axis;
}

#ifndef JIT
static int packed_cuda_copy_width(const Var* input) {
    const int rank = input->shape.size();
    const int width = 16 / input->dsize();
    if (width < 2 || input->storage_stride(rank-1) != 1
            || input->shape[rank-1] % width || input->storage_offset_bytes % 16)
        return 1;
    // An aligned base only suffices if every row begins at a vector boundary.
    for (int axis = 0; axis < rank-1; ++axis)
        if (input->shape[axis] > 1 && input->storage_stride(axis) % width)
            return 1;
    return width;
}

static int tiled_cuda_transpose_axis(const Var* input) {
    const int last = input->shape.size() - 1;
    // Complex types have constructors and cannot be declared in the shared tile.
    if (input->storage_stride(last) <= 1 || input->dtype().is_complex())
        return -1;
    int unit_axis = -1;
    for (int axis = last - 1; axis >= 0; --axis) {
        const auto stride = input->storage_stride(axis);
        if (stride <= 0) return -1;
        if (stride == 1 && (unit_axis < 0 || input->shape[unit_axis] == 1))
            unit_axis = axis;
    }
    // Large batches of narrow, densely packed matrices can leave most lanes
    // idle on every load. The scalar JIT already vectorizes their stores.
    if (unit_axis >= 0 && input->num >= (1 << 18) && input->storage_stride(last) < 16) {
        int64 columns;
        transpose_column_begin(input, unit_axis, columns);
        if (input->storage_stride(last) == columns) return -1;
    }
    return unit_axis;
}

ContiguousOp::ContiguousOp(Var* input) : x(input) {
    set_flag(OpFlags::_cpu);
    set_flag(OpFlags::_cuda);
    set_flag(OpFlags::_manual_set_vnbb);
    if (x->is_contiguous()) { forward(x); return; }
    const auto backend = x->placement.explicit_backend
        ? x->placement.device.backend : execution_target_backend();
    if (backend == BackendId::Cuda && x->num > 0 && x->shape.size() >= 2) {
        cuda_copy_width = packed_cuda_copy_width(x);
        if (cuda_copy_width == 1)
            cuda_transpose_axis = tiled_cuda_transpose_axis(x);
    }
    // Keep transpose candidates eligible for fusion. The executor selects
    // the tiled launch when this copy is alone in its segment.
    set_type(cuda_copy_width > 1 ? OpType::other : OpType::element);
    y = create_output(nullptr, x->dtype());
}
void ContiguousOp::infer_shape() { y->set_shape(x->shape); }
VarPtr ContiguousOp::grad(Var*, Var* dout, Var*, int) { return dout; }
void ContiguousOp::jit_prepare(JK& jk) {
    jk << "«Tx:" << x->dtype() << "«DIM=" << JK::hex(x->shape.size());
    // Which axes move the physical index at all. A view's zero-stride axes
    // contribute nothing to the offset, so saying so in the key lets the copy
    // skip them instead of dividing them away once per element; an all-zero
    // pattern (a scalar broadcast) then reads element 0 for every i. There is
    // deliberately no `offset = i` shortcut here, unlike the elementwise ops:
    // this kernel may be handed a view whose every stride is zero, where `i`
    // would index past the four bytes the source actually owns.
    jk << "«XSMASK=" << JK::hex(x->stride_pattern());
    jk << "«COPY_WIDTH=" << JK::hex(cuda_copy_width);
}
#else
#if defined(JIT_cuda) && COPY_WIDTH > 1
namespace {
struct ContiguousCopy {
    index_t shape[DIM];
    index_t stride[DIM];
    uint32 reciprocal[DIM];
    index_t vectors;
};

template<bool aligned>
__global__ void packed_contiguous(const Tx* __restrict__ input,
                                  Tx* __restrict__ output, ContiguousCopy copy) {
    const index_t step = (index_t)gridDim.x * blockDim.x;
    for (index_t vector = (index_t)blockIdx.x * blockDim.x + threadIdx.x;
            vector < copy.vectors; vector += step) {
        index_t index = vector, offset = 0;
        #pragma unroll
        for (int axis = DIM-1; axis > 0; --axis) {
            if constexpr (sizeof(index_t) == sizeof(uint32)) {
                const uint32 extent = copy.shape[axis];
                const uint32 position = index;
                uint32 quotient = extent == 1 ? position : __umulhi(position, copy.reciprocal[axis]);
                uint32 remainder = position - quotient * extent;
                if (remainder >= extent) {
                    ++quotient;
                    remainder -= extent;
                }
                offset += index_t(remainder) * copy.stride[axis];
                index = index_t(quotient);
            } else {
                offset += (index % copy.shape[axis]) * copy.stride[axis];
                index /= copy.shape[axis];
            }
        }
        offset += index * copy.stride[0];
        const Tx* src = input + offset;
        Tx* dst = output + vector * COPY_WIDTH;
        if constexpr (aligned) {
            *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(src);
        } else {
            #pragma unroll
            for (int lane = 0; lane < COPY_WIDTH; ++lane)
                dst[lane] = src[lane];
        }
    }
}
void launch_packed_contiguous(Var* x, Var* y) {
    auto* __restrict__ xp = x->ptr<Tx>();
    auto* __restrict__ yp = y->ptr<Tx>();
    ContiguousCopy copy{};
    copy.vectors = y->num / COPY_WIDTH;
    if (!copy.vectors) return;
    for (int axis = 0; axis < DIM; ++axis) {
        copy.shape[axis] = x->shape[axis];
        copy.stride[axis] = x->storage_stride(axis);
    }
    copy.shape[DIM-1] /= COPY_WIDTH;
    copy.stride[DIM-1] = COPY_WIDTH;
    if constexpr (sizeof(index_t) == sizeof(uint32)) {
        for (int axis = 1; axis < DIM; ++axis)
            copy.reciprocal[axis] = copy.shape[axis] == 1 ? 0
                : (uint64(1) << 32) / copy.shape[axis];
    }
    constexpr int threads = 256;
    const int blocks = std::min<index_t>((copy.vectors - 1) / threads + 1, 4096);
    // Foreign storage may publish a base pointer that is already offset.
    if ((((size_t)xp | (size_t)yp) & 15) == 0)
        packed_contiguous<true><<<blocks, threads>>>(xp, yp, copy);
    else
        packed_contiguous<false><<<blocks, threads>>>(xp, yp, copy);
}
} // namespace
#endif

#if defined(JIT_cuda) && defined(TILED_TRANSPOSE) && !defined(JIT_fuser)
namespace {
constexpr int transpose_tile_size = 32;
constexpr int transpose_block_rows = 8;

struct TransposeCopy {
    index_t rows, columns;
    index_t input_row_stride, output_column_stride;
    index_t column_tiles, matrix_tiles, tiles;
    int batch_dims;
    index_t batch_shape[DIM];
    index_t input_batch_stride[DIM];
    index_t output_batch_stride[DIM];
};

__global__ void tiled_contiguous(const Tx* __restrict__ input,
                                 Tx* __restrict__ output, TransposeCopy copy) {
    // Give subword elements a full 32-bit bank word. The extra column
    // avoids bank conflicts when the warp reads down the tile.
    struct alignas(sizeof(Tx) < 4 ? 4 : alignof(Tx)) TileElement {
        Tx value;
    };
    __shared__ TileElement tile[transpose_tile_size][transpose_tile_size + 1];
    const index_t step = gridDim.x;
    for (index_t block = blockIdx.x; ; block += step) {
        const index_t matrix_tile = block % copy.matrix_tiles;
        const index_t row = (matrix_tile / copy.column_tiles) * transpose_tile_size;
        const index_t column = (matrix_tile % copy.column_tiles) * transpose_tile_size;
        index_t batch = block / copy.matrix_tiles;
        index_t input_base = 0, output_base = 0;
        for (int axis = 0; axis < copy.batch_dims; ++axis) {
            index_t coord = batch;
            if (axis + 1 < copy.batch_dims) {
                coord %= copy.batch_shape[axis];
                batch /= copy.batch_shape[axis];
            }
            input_base += coord * copy.input_batch_stride[axis];
            output_base += coord * copy.output_batch_stride[axis];
        }
        #pragma unroll
        for (int j = threadIdx.y; j < transpose_tile_size; j += transpose_block_rows) {
            const index_t r = row + j, c = column + threadIdx.x;
            if (r < copy.rows && c < copy.columns)
                tile[j][threadIdx.x].value = input[input_base + r * copy.input_row_stride + c];
        }
        __syncthreads();
        #pragma unroll
        for (int j = threadIdx.y; j < transpose_tile_size; j += transpose_block_rows) {
            const index_t r = row + threadIdx.x, c = column + j;
            if (r < copy.rows && c < copy.columns)
                output[output_base + c * copy.output_column_stride + r] = tile[threadIdx.x][j].value;
        }
        if (copy.tiles - block <= step) break;
        __syncthreads();
    }
}

void launch_tiled_contiguous(Var* x, Var* y, int unit_axis) {
    if (!y->num) return;
    TransposeCopy copy{};
    int64 columns;
    const int column_begin = transpose_column_begin(x, unit_axis, columns);
    int row_begin = DIM - 1;
    copy.columns = columns;
    copy.rows = x->shape[row_begin];
    const int64 row_stride = x->storage_stride(row_begin);
    copy.input_row_stride = row_stride;
    // Fold adjacent axes with consecutive row addresses. Together with
    // transpose_column_begin this treats H and W as one matrix dimension.
    while (row_begin > unit_axis + 1) {
        const int axis = row_begin - 1;
        const int64 stride = x->storage_stride(axis);
        if (x->shape[axis] != 1 && (stride / row_stride != copy.rows || stride % row_stride)) break;
        copy.rows *= x->shape[axis];
        row_begin = axis;
    }
    index_t output_stride = 1;
    for (int axis = DIM - 1; axis >= 0; --axis) {
        const index_t size = x->shape[axis];
        if (axis == unit_axis) copy.output_column_stride = output_stride;
        if (size > 1 && (axis < column_begin || (axis > unit_axis && axis < row_begin))) {
            const int batch_axis = copy.batch_dims++;
            copy.batch_shape[batch_axis] = size;
            copy.input_batch_stride[batch_axis] = x->storage_stride(axis);
            copy.output_batch_stride[batch_axis] = output_stride;
        }
        output_stride *= size;
    }
    copy.column_tiles = (copy.columns - 1) / transpose_tile_size + 1;
    const index_t row_tiles = (copy.rows - 1) / transpose_tile_size + 1;
    copy.matrix_tiles = row_tiles * copy.column_tiles;
    copy.tiles = copy.matrix_tiles * (y->num / copy.rows / copy.columns);
    const int blocks = std::min<index_t>(copy.tiles, 65535);
    const dim3 threads(transpose_tile_size, transpose_block_rows);
    // Var::ptr starts at the first logical element of a view.
    tiled_contiguous<<<blocks, threads>>>(x->ptr<Tx>(), y->ptr<Tx>(), copy);
}
} // namespace
#endif

void ContiguousOp::jit_run() {
    @if(@is_def(JIT_cuda) && !@is_def(JIT_fuser) && (COPY_WIDTH > 1 || @is_def(TILED_TRANSPOSE)),
        @if(COPY_WIDTH > 1,
            launch_packed_contiguous(x, y);
        ,
            launch_tiled_contiguous(x, y, cuda_transpose_axis);
        )
    ,
        auto* __restrict__ xp = x->ptr<Tx>();
        auto* __restrict__ yp = y->ptr<Tx>();
        @for(d, 0, DIM, index_t xshape@d = x->shape[@d];)
        @for(d, 0, DIM, index_t xstride@d = x->storage_stride(@d);)
        // The product of the higher axes' shapes gives one division per axis.
        index_t xabove@{DIM-1} = 1;
        @for(d, DIM-2, -1, -1, index_t xabove@d = xabove@{d+1} * xshape@{d+1};)
        index_t num = y->num;
        for (index_t i=0; i<num; ++i) {
            index_t offset = 0;
            @for(d, 0, DIM,
                @if(XSMASK>>d&1,
                    offset += @if(d, (i / xabove@d % xshape@d), (i / xabove@d)) * xstride@d;
                )
            )
            yp[i] = xp[offset];
        }
    )
}
#endif
}
