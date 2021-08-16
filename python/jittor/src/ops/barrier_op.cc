// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: 
//     Guowei Yang <471184555@qq.com>
//     Dun Liang <randonlang@gmail.com>. 
// All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "var.h"
#include "ops/array_op.h"
#include "ops/op_register.h"
#include "ops/barrier_op.h"

namespace jittor {

BarrierOp::BarrierOp(vector<Var*>&& x) {
    flags.set(NodeFlags::_cpu);
    flags.set(NodeFlags::_cuda);
    flags.set(NodeFlags::_grads);
    for (uint i = 0; i < x.size(); ++i) 
        create_output(nullptr, x[i]->dtype());
}

void BarrierOp::grads(Var** douts, VarPtr* dins) {
    int n = inputs().size();
    for (int i=0; i<n; i++)
        dins[i] = douts[i];
}

void BarrierOp::infer_shape() {
    auto yiter = outputs().begin();
    for (Var* x : inputs()) {
        (*yiter)->set_shape(x->shape);
        (*yiter)->share_with(x);
        yiter++;
    }
}

} // jittor