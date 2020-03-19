// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: 
//     Guoye Yang <498731903@qq.com>
//     Dun Liang <randonlang@gmail.com>. 
// All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "var.h"
#include "ops/array_op.h"
#include "ops/op_register.h"
#include "ops/reshape_op.h"

namespace jittor {

static auto make_reshape = get_op_info("reshape")
    .get_constructor<VarPtr, Var*, NanoVector>();

ReshapeOp::ReshapeOp(Var* x, NanoVector shape) : x(x), shape(shape) {
    flags.set(NodeFlags::_cpu);
    flags.set(NodeFlags::_cuda);
    y = create_output(nullptr, x->dtype());
    ASSERT(shape.size() > 0) << "input target shape of reshape can't be empty.";
}

VarPtr ReshapeOp::grad(Var* out, Var* dout, Var* v, int v_index) {
    return make_reshape(dout, x->shape);
}

void ReshapeOp::infer_shape() {
    size_t uncertain_dim = 0;
    int64_t y_items = 1;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] == -1) {
            ++uncertain_dim;
        } else
            y_items *= shape[i];
    }
    ASSERT(uncertain_dim <= 1) << "max number of -1 is 1, but get" << uncertain_dim << ".";
    int64_t x_items = x->num;
    auto yshape = shape;
    if (uncertain_dim == 0) {
        ASSERT(x_items == y_items) << "reshape shape is invalid for input of size " << x_items;
    } else {
        ASSERT(x_items % y_items == 0) << "reshape shape is invalid for input of size " << x_items;
        uncertain_dim = x_items / y_items;
        yshape.clear();
        for (auto a : shape)
            yshape.push_back(a==-1 ? uncertain_dim : a);
    }
    y->set_shape(yshape);
    y->share_with(x);
}
} // jittor