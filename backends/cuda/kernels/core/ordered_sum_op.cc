#include "ops/composite/ordered_sum_op.h"
#include "core/var.h"
#include "helper_cuda.h"
#include "ordered_nonlast_sum.h"

namespace jittor {
#ifdef JIT_cuda
void OrderedSumOp::jit_run() {
    const int rank=x->shape.size();
#if LAST_AXIS
    const int columns=x->shape[rank-1];
    const int rows=y->num;
    jittor_ordered_sum::launch_last_fp32(x->ptr<float>(),y->ptr<float>(),rows,columns);
#else
    const int rows=x->shape[rank-2], columns=x->shape[rank-1];
    const int outputs=y->num;
    if (columns==1024 && x->num==76LL*1024 && outputs==1024) {
        // supports_ordered_sum admits only full-prefix/zero-offset/dense x.
        // Validate actual final pointers; metadata admission never implies a
        // permission to dereference an unaligned float4 or to copy to host.
        USER_CHECK(x->is_contiguous() && x->storage_offset_bytes==0 &&
            !(reinterpret_cast<uintptr_t>(x->ptr<float>()) & 15u) &&
            !(reinterpret_cast<uintptr_t>(y->ptr<float>()) & 15u))
            << "leading76c1024 requires actual aligned dense CUDA storage";
        jittor_ordered_sum::leading76c1024_fp32<<<8,dim3(32,4)>>>(x->ptr<float>(),y->ptr<float>());
    } else {
        int blocks=(outputs+255)/256;
        if (blocks>4096) blocks=4096;
        jittor_ordered_sum::nonlast_fp32<<<blocks,256>>>(
            x->ptr<float>(),y->ptr<float>(),outputs,rows,columns);
    }
#endif
    checkCudaErrors(cudaGetLastError());
}
#endif
} // namespace jittor
