// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: 
//     Dun Liang <randonlang@gmail.com>. 
// All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "var.h"
#include "ops/array_op.h"
#include "ops/op_register.h"
#include "ops/tape_op.h"

namespace jittor {

static auto make_tape = get_op_info("tape")
    .get_constructor<VarPtr, Var*>();

TapeOp::TapeOp(Var* x) : tapes(nullptr) {
    flags.set(NodeFlags::_cpu);
    flags.set(NodeFlags::_cuda);
    auto y = create_output(nullptr, x->dtype());
    if (x->name.ptr)
        y->name = x->name;
}

TapeOp::~TapeOp() {
    if (tapes) {
        if (! --tapes->ref) {
            tapes->_inputs.clear();
            tapes->_outputs.clear();
            delete tapes;
        }
    }
}

VarPtr TapeOp::grad(Var* out, Var* dout, Var* v, int v_index) {
    return dout;
}

void TapeOp::infer_shape() {
    auto x = inputs().front();
    auto y = outputs().front();
    y->set_shape(x->shape);
    y->share_with(x);
}

void Tapes::grads(Var** douts, VarPtr* dins) {
    callback.func(_outputs.size(), douts, _inputs.size(), dins);
}

} // jittor