// Copyright (c) 2026 Jittor. All Rights Reserved.
// This file is subject to the terms in LICENSE.txt.
#include "core/var.h"
#include "nccl_wait_many_op.h"
#include "nccl_wrapper.h"
#include "ops/op_register.h"

namespace jittor {
#ifndef JIT
NcclWaitManyOp::NcclWaitManyOp(vector<Var*>&& inputs) : tensor_count(inputs.size()) {
    USER_CHECK(tensor_count) << "nccl_wait_many requires at least one tensor";
    set_flag(OpFlags::_cpu, 0);
    set_flag(OpFlags::_cuda, 1);
    set_flag(OpFlags::_manual_set_vnbb);
    // The maker appends these variadic inputs after construction.
    for (auto* value : inputs) create_output(nullptr, value->dtype());
}

void NcclWaitManyOp::infer_shape() {
    for (int i=0; i<tensor_count; ++i) {
        output(i)->set_shape(input(i)->shape);
        output(i)->share_with(input(i));
    }
}

void NcclWaitManyOp::jit_prepare(JK& jk) { jk << "«N:" << tensor_count; }

VarPtr NcclWaitManyOp::grad(Var* out, Var* dout, Var* v, int v_index) {
    if (v_index < 0 || v_index >= tensor_count || out != output(v_index)) return nullptr;
    // Even a constant cotangent must not start another collective while the
    // forward's communication still owes its join.
    static auto clone = op_constructor<VarPtr, Var*>("clone");
    auto result = clone(dout);
    auto* producer = result->input();
    vector<Node*> dependency{out};
    producer->add_inputs(dependency);
    producer->_inputs.back().reverse().index = -1;
    return result;
}
#else
#ifdef JIT_cuda
void NcclWaitManyOp::jit_run() {
    // This is executable ordering, even though the outputs alias the inputs.
    // Treating it as a storage-only view would silently drop the stream join.
    nccl_comm_wait();
}
#endif
#endif
} // namespace jittor
