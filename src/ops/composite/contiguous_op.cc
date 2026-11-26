#include <algorithm>
#include "ops/composite/contiguous_op.h"
#include "core/var.h"

namespace jittor {
#ifndef JIT
static int packed_cuda_copy_width(Var* input) {
    const auto backend = input->placement.explicit_backend
        ? input->placement.device.backend : execution_target_backend();
    if (backend != BackendId::Cuda || input->num <= 0)
        return 1;
    const int rank = input->shape.size();
    const int width = 16 / input->dsize();
    if (rank < 2 || width < 2 || input->storage_stride(rank-1) != 1
            || input->shape[rank-1] % width || input->storage_offset_bytes % 16)
        return 1;
    // An aligned base only suffices if every row begins at a vector boundary.
    for (int axis = 0; axis < rank-1; ++axis)
        if (input->shape[axis] > 1 && input->storage_stride(axis) % width)
            return 1;
    return width;
}

ContiguousOp::ContiguousOp(Var* input) : x(input) {
    set_flag(OpFlags::_cpu);
    set_flag(OpFlags::_cuda);
    set_flag(OpFlags::_manual_set_vnbb);
    if (x->is_contiguous()) { forward(x); return; }
    cuda_copy_width = packed_cuda_copy_width(x);
    // A packed kernel transfers a whole vector per thread. The scalar form
    // can still be fused with its consumers on the other layouts.
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
            offset += (index % copy.shape[axis]) * copy.stride[axis];
            index /= copy.shape[axis];
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
    ContiguousCopy copy;
    copy.vectors = y->num / COPY_WIDTH;
    if (!copy.vectors) return;
    for (int axis = 0; axis < DIM; ++axis) {
        copy.shape[axis] = x->shape[axis];
        copy.stride[axis] = x->storage_stride(axis);
    }
    copy.shape[DIM-1] /= COPY_WIDTH;
    copy.stride[DIM-1] = COPY_WIDTH;
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

void ContiguousOp::jit_run() {
    @if(@is_def(JIT_cuda) && COPY_WIDTH > 1,
        launch_packed_contiguous(x, y);
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
