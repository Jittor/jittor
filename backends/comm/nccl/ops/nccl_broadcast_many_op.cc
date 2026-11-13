// Copyright (c) 2026 Jittor. All Rights Reserved.
// This file is subject to the terms in LICENSE.txt.
#include "core/var.h"
#include "nccl_broadcast_many_op.h"
#include "nccl_wrapper.h"
#include "ops/op_register.h"

namespace jittor {
#ifndef JIT
NcclBroadcastManyOp::NcclBroadcastManyOp(vector<Var*>&& inputs, int root, int group_id)
    : tensor_count(inputs.size()), root(root), group_id(group_id) {
    USER_CHECK(tensor_count) << "nccl_broadcast_many requires at least one tensor";
    set_flag(OpFlags::_cpu, 0);
    set_flag(OpFlags::_cuda, 1);
    for (auto* value : inputs) create_output(nullptr, value->dtype());
}

void NcclBroadcastManyOp::infer_shape() {
    for (int i=0; i<tensor_count; ++i) output(i)->set_shape(input(i)->shape);
}

void NcclBroadcastManyOp::jit_prepare(JK& jk) { jk << "«N:" << tensor_count; }

VarPtr NcclBroadcastManyOp::grad(Var* out, Var* dout, Var* v, int v_index) {
    if (v_index < 0 || v_index >= tensor_count || out != output(v_index)) return nullptr;
    static auto reduce = op_constructor<VarPtr, Var*, int, int>("nccl_reduce");
    return reduce(dout, root, group_id);
}
#else
#ifdef JIT_cuda
void NcclBroadcastManyOp::jit_run() {
    for (int i=0; i<tensor_count; ++i) nccl_dtype(input(i)->dtype());
    // The group begins and ends on the executor thread, including IR replay.
    nccl_bucket_begin(false);
    try {
        for (int i=0; i<tensor_count; ++i) {
            auto* x = input(i);
            auto* y = output(i);
            if (!x->num) continue;
            auto stream = nccl_stream_begin();
            checkCudaErrors(ncclBroadcast(x->mem_ptr, y->mem_ptr, x->num,
                nccl_dtype(x->dtype()), root, nccl_process_group_comm(group_id), stream));
            nccl_stream_end(x, y);
        }
    } catch (...) {
        nccl_bucket_end();
        throw;
    }
    nccl_bucket_end();
}
#endif
#endif
} // namespace jittor
