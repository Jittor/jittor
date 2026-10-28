// ***************************************************************
// Copyright (c) 2026 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "ops/composite/mapped_packed_matmul_op.h"
#include "core/var.h"
#include "runtime/dispatch_context.h"
#include "runtime/device.h"

namespace jittor {

#ifndef JIT

DECLARE_FLAG(bool, no_grad);

MappedPackedMatmulOp::MappedPackedMatmulOp(Var* a, Var* packed_weight,
        int64 in_features, int64 out_features)
    : a(a), b(packed_weight), in_features(in_features), out_features(out_features) {
    USER_CHECK(no_grad)
        << "mapped_packed_matmul requires no_grad; training is unsupported";
    USER_CHECK(a->dtype() == ns_float16 && b->dtype() == ns_float16)
        << "mapped_packed_matmul requires float16 operands, but got activation:"
        << a->dtype() << "packed weight:" << b->dtype();
    USER_CHECK(runtime_use_cuda())
        << "mapped_packed_matmul requires the ACL device backend";
    const auto target = query_dispatch_context({a, b});
    USER_CHECK(target.backend == "acl" || target.backend == "acl_legacy")
        << "mapped_packed_matmul requires the ACL device backend, but got" << target.backend;
    set_flag(OpFlags::_cuda, 1);
    set_flag(OpFlags::_cpu, 0);
    set_flag(OpFlags::_manual_set_vnbb);
    c = create_output(nullptr, a->dtype());
}

VarPtr MappedPackedMatmulOp::grad(Var* out, Var* dout, Var* v, int v_index) {
    USER_ERROR << "mapped_packed_matmul is inference-only and has no backward";
    return nullptr;
}

void MappedPackedMatmulOp::infer_shape() {
    USER_CHECK(in_features > 0 && out_features > 0)
        << "mapped_packed_matmul requires positive input and output features, but got"
        << in_features << "and" << out_features;
    const auto rank = a->shape.size();
    USER_CHECK(rank >= 1 && a->shape[rank-1] == in_features)
        << "mapped_packed_matmul input last dimension must equal in_features:"
        << in_features << "but got" << a->shape;
    USER_CHECK(b->shape.size() == 4
            && b->shape[0] == (in_features - 1) / 16 + 1
            && b->shape[1] == (out_features - 1) / 16 + 1
            && b->shape[2] == 16 && b->shape[3] == 16)
        << "mapped_packed_matmul packed weight shape does not match feature dimensions:"
        << b->shape << "for" << in_features << "and" << out_features;
    NanoVector shape;
    for (int axis = 0; axis + 1 < rank; ++axis)
        shape.push_back_check_overflow(a->shape[axis]);
    shape.push_back_check_overflow(out_features);
    c->set_shape(shape);
}

void MappedPackedMatmulOp::jit_prepare(JK& jk) {
    // Keep mapped execution visible to the operator profiler, as the ordinary
    // mapped product does. All dimensions are read from the node at execution.
    jk << "«T:" << a->dtype();
}

#else
void MappedPackedMatmulOp::jit_run() {
    USER_ERROR << "mapped_packed_matmul is only available through a mapped backend";
}
#endif

} // jittor
