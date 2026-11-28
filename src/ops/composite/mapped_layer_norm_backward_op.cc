// ***************************************************************
// Copyright (c) 2026 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "ops/composite/mapped_layer_norm_backward_op.h"
#include "core/var.h"

namespace jittor {

#ifndef JIT

MappedLayerNormBackwardOp::MappedLayerNormBackwardOp(
    Var* dout, Var* x, Var* mean, Var* rstd,
    Var* weight, Var* bias,
    int normalized_ndim, double eps)
    : dout(dout), x(x), mean(mean), rstd(rstd), weight(weight), bias(bias),
      dx(nullptr), dw(nullptr), db(nullptr),
      normalized_ndim(normalized_ndim), eps(eps) {
    set_flag(OpFlags::_cuda, 1);
    set_flag(OpFlags::_cpu, 0);
    set_flag(OpFlags::_manual_set_vnbb);
    dx = create_output(nullptr, x->dtype());
    dw = create_output(nullptr, weight->dtype());
    db = create_output(nullptr, bias->dtype());
}

void MappedLayerNormBackwardOp::infer_shape() {
    dx->set_shape(x->shape);
    dw->set_shape(weight->shape);
    db->set_shape(bias->shape);
}

VarPtr MappedLayerNormBackwardOp::grad(Var* out, Var* dout, Var* v, int v_index) {
    USER_ERROR << "mapped_layer_norm_backward does not support second-order differentiation";
    return nullptr;
}

void MappedLayerNormBackwardOp::jit_prepare(JK& jk) {
    jk << "«T:" << dout->dtype() << "«ND:" << normalized_ndim;
}

#else
void MappedLayerNormBackwardOp::jit_run() {
    USER_ERROR << "mapped_layer_norm_backward is only available through a mapped backend";
}
#endif

} // jittor
