// ***************************************************************
// Copyright (c) 2026 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "ops/composite/mapped_unary_grad_op.h"
#include "ops/composite/mapped_unary_op.h"
#include "core/var.h"
#include "ops/op_register.h"

namespace jittor {

#ifndef JIT

static auto make_mapped_unary_grad_self =
    op_constructor<VarPtr, Var*, Var*, int, double>("mapped_unary_grad");

MappedUnaryGradOp::MappedUnaryGradOp(Var* dout, Var* x, int kind, double param0)
    : dout(dout), x(x), dx(nullptr), kind(kind), param0(param0) {
    USER_CHECK(kind == MappedUnaryOp::kRelu || kind == MappedUnaryOp::kGelu)
        << "mapped_unary_grad supports only kRelu and kGelu, got" << kind;
    set_flag(OpFlags::_cuda, 1);
    set_flag(OpFlags::_cpu, 0);
    set_flag(OpFlags::_manual_set_vnbb);
    if (kind == MappedUnaryOp::kRelu)
        x->set_flag(VarFlags::_needed_by_backward);
    dx = create_output(nullptr, dout->dtype());
}

void MappedUnaryGradOp::infer_shape() {
    dx->set_shape(x->shape);
}

VarPtr MappedUnaryGradOp::grad(Var* out, Var* ddout, Var* v, int v_index) {
    if (kind == MappedUnaryOp::kRelu) {
        if (v_index == 0)
            return make_mapped_unary_grad_self(ddout, x, kind, 0.0);
        return nullptr;
    }
    USER_ERROR << "mapped_unary_grad gelu does not support second-order differentiation";
    return nullptr;
}

void MappedUnaryGradOp::jit_prepare(JK& jk) {
    jk << "«K:" << kind << "«T:" << dout->dtype();
}

#else
void MappedUnaryGradOp::jit_run() {
    USER_ERROR << "mapped_unary_grad is only available through a mapped backend";
}
#endif

} // jittor
