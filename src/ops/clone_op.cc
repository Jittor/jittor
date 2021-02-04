// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: 
//     Dun Liang <randonlang@gmail.com>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "var.h"
#include "ops/array_op.h"
#include "ops/op_register.h"
#include "ops/clone_op.h"

namespace jittor {

static auto make_clone = get_op_info("clone")
    .get_constructor<VarPtr, Var*>();

CloneOp::CloneOp(Var* x) : x(x) {
    flags.set(NodeFlags::_cpu);
    flags.set(NodeFlags::_cuda);
    y = create_output(nullptr, x->dtype());
    if (x->name.ptr)
        y->name = x->name;
}

VarPtr CloneOp::grad(Var* out, Var* dout, Var* v, int v_index) {
    return make_clone(dout);
}

void CloneOp::infer_shape() {
    y->set_shape(x->shape);
    y->share_with(x);
}

VarPtr detach(Var* x) {
    auto y = make_clone(x);
    y->input()->set_stop_grad();
    return y;
}

} // jittor