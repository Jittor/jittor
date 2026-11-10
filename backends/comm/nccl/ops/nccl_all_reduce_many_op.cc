// Copyright (c) 2026 Jittor. All Rights Reserved.
// This file is subject to the terms in LICENSE.txt.
#include "core/var.h"
#include "nccl_all_reduce_many_op.h"
#include "nccl_wrapper.h"
#include "ops/op_register.h"

namespace jittor {

#ifndef JIT
static auto make_all_reduce = op_constructor<VarPtr, Var*, int>("nccl_all_reduce");
static auto make_many = op_constructor<vector<VarPtr>, vector<Var*>&&, int, bool>("nccl_all_reduce_many");

NcclAllReduceManyOp::NcclAllReduceManyOp(vector<Var*>&& inputs, int group_id, bool mean)
    : tensor_count(inputs.size()), group_id(group_id), mean(mean) {
    USER_CHECK(tensor_count) << "nccl_all_reduce_many requires at least one tensor";
    set_flag(OpFlags::_cpu, 0);
    set_flag(OpFlags::_cuda, 1);
    // The maker registers variadic edges after construction, so leave the
    // argument intact. Read Op edges during execution: cached Var* vectors
    // would also miss the input replacements performed by IR replay.
    for (auto* x : inputs) {
        USER_CHECK(!mean || x->dtype() == ns_float32) << "grouped native mean requires float32";
        create_output(nullptr, x->dtype());
    }
}

void NcclAllReduceManyOp::infer_shape() {
    for (int i=0; i<tensor_count; ++i) output(i)->set_shape(input(i)->shape);
}

void NcclAllReduceManyOp::jit_prepare(JK& jk) {
    jk << "«N:" << tensor_count << "«MEAN:" << (int)mean;
}

VarPtr NcclAllReduceManyOp::grad(Var* out, Var* dout, Var* v, int v_index) {
    if (v_index < 0 || v_index >= tensor_count || out != output(v_index)) return nullptr;
    if (mean) {
        vector<Var*> inputs{dout};
        auto values = make_many(move(inputs), group_id, true);
        return move(values[0]);
    }
    return make_all_reduce(dout, group_id);
}

#else
#ifdef JIT_cuda
void NcclAllReduceManyOp::jit_run() {
    // Validate before opening a group; an unsupported dtype must not leave a
    // half-open group for the next unrelated collective.
    for (int i=0; i<tensor_count; ++i) nccl_dtype(input(i)->dtype());
    nccl_bucket_begin(false);
    try {
        for (int i=0; i<tensor_count; ++i) {
            Var* x = input(i);
            Var* y = output(i);
            if (!x->num) continue;
            auto stream = nccl_stream_begin();
            checkCudaErrors(ncclAllReduce(
                x->mem_ptr, y->mem_ptr, x->num,
                nccl_dtype(x->dtype()), mean ? ncclAvg : ncclSum,
                nccl_process_group_comm(group_id), stream));
            nccl_stream_end(x, y);
        }
    } catch (...) {
        nccl_bucket_end();
        throw;
    }
    // Results become readable only after the native group's single join.
    nccl_bucket_end();
}
#endif
#endif

} // namespace jittor
