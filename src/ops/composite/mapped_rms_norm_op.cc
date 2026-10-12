// ***************************************************************
// Copyright (c) 2026 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <cmath>
#include "ops/composite/mapped_rms_norm_op.h"
#include "core/var.h"
#include "runtime/dispatch_context.h"
#include "runtime/device.h"

namespace jittor {

#ifndef JIT

DECLARE_FLAG(bool, no_grad);

MappedRmsNormOp::MappedRmsNormOp(Var* x, Var* weight, double eps)
    : x(x), weight(weight), eps(eps) {
    USER_CHECK(no_grad)
        << "mapped_rms_norm requires no_grad; use the differentiable RMSNorm path for training";
    USER_CHECK(x->dtype() == ns_float16 && weight->dtype() == ns_float16)
        << "mapped_rms_norm requires float16 input and weight; use the original RMSNorm path for other dtypes, but got"
        << x->dtype() << "and" << weight->dtype();
    USER_CHECK(std::isfinite(eps) && eps > 0)
        << "mapped_rms_norm epsilon must be finite and positive, but got" << eps;
    USER_CHECK(runtime_use_cuda()) << "mapped_rms_norm requires the ACL device backend";
    const auto target = query_dispatch_context({x, weight});
    USER_CHECK(target.backend == "acl" || target.backend == "acl_legacy")
        << "mapped_rms_norm requires the ACL device backend, but got" << target.backend;
    set_flag(OpFlags::_cuda);
    set_flag(OpFlags::_cpu, 0);
    set_flag(OpFlags::_manual_set_vnbb);
    y = create_output(nullptr, x->dtype());
}

VarPtr MappedRmsNormOp::grad(Var* out, Var* dout, Var* v, int v_index) {
    USER_ERROR << "mapped_rms_norm is inference-only and has no backward";
    return nullptr;
}

void MappedRmsNormOp::infer_shape() {
    const auto rank = x->shape.size();
    USER_CHECK(rank >= 1) << "mapped_rms_norm input must have at least one dimension";
    for (int axis = 0; axis < rank; ++axis)
        USER_CHECK(x->shape[axis] > 0)
            << "mapped_rms_norm requires positive input dimensions, but got" << x->shape;
    USER_CHECK(weight->shape.size() == 1 && weight->shape[0] == x->shape[rank-1])
        << "mapped_rms_norm weight must match the input's final dimension, but got"
        << weight->shape << "and" << x->shape;
    y->set_shape(x->shape);
    NanoVector reduced;
    for (int axis = 0; axis + 1 < rank; ++axis)
        reduced.push_back_check_overflow(x->shape[axis]);
    reduced.push_back_check_overflow(1);
    // An independent Var, with no producer edge back to this op: ownership
    // cannot form a cycle. Replacing it drops any old allocation when a
    // dynamic input changes the scratch extent instead of resizing old bytes.
    if (!rstd_scratch || !(rstd_scratch->shape == reduced))
        rstd_scratch = VarPtr(reduced, ns_float32);
}

void MappedRmsNormOp::jit_prepare(JK& jk) {
    // Dimensions and epsilon remain node data. A nonempty key keeps the
    // mapped launch visible to the same profiler as other native operators.
    jk << "«Tx:" << x->dtype() << "«Tw:" << weight->dtype();
}

#else
void MappedRmsNormOp::jit_run() {
    USER_ERROR << "mapped_rms_norm is only available through a mapped backend";
}
#endif

} // jittor
