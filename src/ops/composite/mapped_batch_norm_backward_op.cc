// ***************************************************************
// Copyright (c) 2026 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "ops/composite/mapped_batch_norm_backward_op.h"
#include "core/var.h"

namespace jittor {

#ifndef JIT

MappedBatchNormBackwardOp::MappedBatchNormBackwardOp(
    Var* dout, Var* x, Var* weight,
    Var* running_mean, Var* running_var,
    Var* save_mean, Var* save_invstd,
    double eps, bool is_train)
    : dout(dout), x(x), weight(weight),
      running_mean(running_mean), running_var(running_var),
      save_mean(save_mean), save_invstd(save_invstd),
      dx(nullptr), dw(nullptr), db(nullptr),
      eps(eps), is_train(is_train) {
    set_flag(OpFlags::_cuda, 1);
    set_flag(OpFlags::_cpu, 0);
    set_flag(OpFlags::_manual_set_vnbb);
    dx = create_output(nullptr, x->dtype());
    dw = create_output(nullptr, weight->dtype());
    db = create_output(nullptr, weight->dtype());
}

void MappedBatchNormBackwardOp::infer_shape() {
    dx->set_shape(x->shape);
    dw->set_shape(weight->shape);
    db->set_shape(weight->shape);
}

VarPtr MappedBatchNormBackwardOp::grad(Var* out, Var* dout, Var* v, int v_index) {
    USER_ERROR << "mapped_batch_norm_backward does not support second-order differentiation";
    return nullptr;
}

void MappedBatchNormBackwardOp::jit_prepare(JK& jk) {
    jk << "«T:" << dout->dtype() << "«TR:" << (is_train ? 1 : 0);
}

#else
void MappedBatchNormBackwardOp::jit_run() {
    USER_ERROR << "mapped_batch_norm_backward is only available through a mapped backend";
}
#endif

} // jittor
