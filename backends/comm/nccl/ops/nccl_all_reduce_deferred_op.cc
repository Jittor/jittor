// Copyright (c) 2026 Jittor. All Rights Reserved.
// This file is subject to the terms in LICENSE.txt.
#include "core/var.h"
#include "nccl_all_reduce_deferred_op.h"
#include "nccl_wrapper.h"
#include "mem/allocator.h"
#include "ops/op_register.h"

namespace jittor {
#ifndef JIT
NcclAllReduceDeferredOp::NcclAllReduceDeferredOp(Var* x, int group_id, bool mean)
    : group_id(group_id), mean(mean) {
    USER_CHECK(!mean || x->dtype() == ns_float32)
        << "deferred native mean requires float32";
    set_flag(OpFlags::_cpu, 0);
    set_flag(OpFlags::_cuda, 1);
    create_output(nullptr, x->dtype());
}

void NcclAllReduceDeferredOp::infer_shape() {
    output(0)->set_shape(input(0)->shape);
}

void NcclAllReduceDeferredOp::jit_prepare(JK& jk) {
    jk << "«Tx:" << input(0)->dtype() << "«MEAN:" << (int)mean;
}

VarPtr NcclAllReduceDeferredOp::grad(Var* out, Var* dout, Var* v, int v_index) {
    if (v_index != 0 || out != output(0)) return nullptr;
    static auto reduce = op_constructor<vector<VarPtr>, vector<Var*>&&, int, bool>("nccl_all_reduce_many");
    vector<Var*> inputs{dout};
    auto values = reduce(move(inputs), group_id, mean);
    return move(values[0]);
}
#else
#ifdef JIT_cuda
void NcclAllReduceDeferredOp::jit_run() {
    // Always use the current Op edges: IR replay can replace the inputs.
    auto* x = input(0);
    auto* y = output(0);
    auto dtype = nccl_dtype(x->dtype());
    if (!x->num) return;
    int device = current_device();
    bool share = x->allocator && y->allocator
        && x->allocator->can_share() && y->allocator->can_share();
    if (share) {
        // A lazy graph may release either Var before the consumer-side join.
        // Keep both allocations reserved, including graph-pool scratch.
        for (auto* value : {x, y}) {
            value->allocator->share_with(value->size, value->allocation);
            CHECK(cuda_side_stream_hold_block(CUDA_COMMUNICATION_STREAM, device,
                value->mem_ptr, value->allocation, value->size, value->allocator));
        }
    }
    auto stream = nccl_stream_begin();
    checkCudaErrors(ncclAllReduce(x->mem_ptr, y->mem_ptr, x->num, dtype,
        mean ? ncclAvg : ncclSum, nccl_process_group_comm(group_id), stream));
    if (share) {
        // No host group is left open. Subsequent buckets use the same ordered
        // stream and append their pins until nccl_wait_many joins all of them.
        cuda_side_stream_defer_join(CUDA_COMMUNICATION_STREAM, device);
    } else {
        nccl_stream_end(x, y);
        nccl_comm_wait();
    }
}
#endif
#endif
} // namespace jittor
