// ***************************************************************
// Copyright (c) 2026 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "ops/composite/mapped_conv2d_backward_op.h"
#include "core/var.h"

namespace jittor {

#ifndef JIT

MappedConv2dBackwardOp::MappedConv2dBackwardOp(
    Var* dout, Var* x, Var* weight,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dil_h, int dil_w,
    int groups, bool has_bias, int cube_math_type)
    : dout(dout), x(x), weight(weight),
      dx(nullptr), dw(nullptr), db(nullptr),
      stride_h(stride_h), stride_w(stride_w),
      pad_h(pad_h), pad_w(pad_w),
      dil_h(dil_h), dil_w(dil_w),
      groups(groups), has_bias(has_bias),
      cube_math_type(cube_math_type) {
    set_flag(OpFlags::_cuda, 1);
    set_flag(OpFlags::_cpu, 0);
    set_flag(OpFlags::_manual_set_vnbb);
    dx = create_output(nullptr, x->dtype());
    dw = create_output(nullptr, weight->dtype());
    if (has_bias)
        db = create_output(nullptr, weight->dtype());
}

void MappedConv2dBackwardOp::infer_shape() {
    dx->set_shape(x->shape);
    dw->set_shape(weight->shape);
    if (db)
        db->set_shape({weight->shape[0]});
}

VarPtr MappedConv2dBackwardOp::grad(Var* out, Var* dout, Var* v, int v_index) {
    USER_ERROR << "mapped_conv2d_backward does not support second-order differentiation";
    return nullptr;
}

void MappedConv2dBackwardOp::jit_prepare(JK& jk) {
    jk << "«T:" << dout->dtype() << "«B:" << (has_bias ? 1 : 0) << "«C:" << cube_math_type;
}

#else
void MappedConv2dBackwardOp::jit_run() {
    USER_ERROR << "mapped_conv2d_backward is only available through a mapped backend";
}
#endif

} // jittor
