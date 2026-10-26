// ***************************************************************
// Copyright (c) 2026 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "ops/composite/mapped_concat_op.h"
#include "core/var.h"
#include "runtime/dispatch_context.h"
#include "runtime/device.h"
#include <limits>

namespace jittor {

#ifndef JIT

DECLARE_FLAG(bool, no_grad);

MappedConcatOp::MappedConcatOp(vector<Var*>&& tensors, int dim)
    : tensors(tensors), dim(dim) {
    // The generated constructor still reads its argument to add graph edges.
    USER_CHECK(no_grad) << "mapped_concat requires no_grad; training is unsupported";
    USER_CHECK(!this->tensors.empty()) << "mapped_concat requires at least one input";
    auto dtype = this->tensors[0]->dtype();
    for (auto* value : this->tensors)
        USER_CHECK(value->dtype() == dtype)
            << "mapped_concat inputs must have the same dtype";
    USER_CHECK(runtime_use_cuda()) << "mapped_concat requires the ACL device backend";
    const auto target = query_dispatch_context(this->tensors);
    USER_CHECK(target.backend == "acl" || target.backend == "acl_legacy")
        << "mapped_concat requires the ACL device backend, but got" << target.backend;
    set_flag(OpFlags::_cuda, 1);
    set_flag(OpFlags::_cpu, 0);
    set_flag(OpFlags::_manual_set_vnbb);
    y = create_output(nullptr, dtype);
}

VarPtr MappedConcatOp::grad(Var* out, Var* dout, Var* v, int v_index) {
    USER_ERROR << "mapped_concat is inference-only and has no backward";
    return nullptr;
}

void MappedConcatOp::infer_shape() {
    const auto& first = tensors[0]->shape;
    const int rank = first.size();
    USER_CHECK(dim >= -rank && dim < rank) << "mapped_concat dim out of range";
    if (dim < 0) dim += rank;
    int64 extent = 0;
    for (auto* value : tensors) {
        const auto& shape = value->shape;
        USER_CHECK(shape.size() == rank)
            << "mapped_concat inputs must have the same rank";
        for (int axis = 0; axis < rank; ++axis)
            USER_CHECK(axis == dim || shape[axis] == first[axis])
                << "mapped_concat inputs must have the same shape outside dim";
        USER_CHECK(shape[dim] >= 0 && extent <= std::numeric_limits<int64>::max() - shape[dim])
            << "mapped_concat output extent overflows";
        extent += shape[dim];
    }
    NanoVector shape;
    for (int axis = 0; axis < rank; ++axis)
        shape.push_back_check_overflow(axis == dim ? extent : first[axis]);
    y->set_shape(shape);
}

void MappedConcatOp::jit_prepare(JK& jk) {
    // Shape, arity and axis live on the node; the mapped runner reads them
    // for each execution, including after a dynamic extent change.
    jk << "«T:" << tensors[0]->dtype();
}

#else
void MappedConcatOp::jit_run() {
    USER_ERROR << "mapped_concat is only available through a mapped backend";
}
#endif

} // jittor
