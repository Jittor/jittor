// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: 
//     Dun Liang <randonlang@gmail.com>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "core/var.h"
#include "ops/composite/array_op.h"
#include "ops/op_register.h"
#include "ops/composite/clone_op.h"

namespace jittor {

static auto make_clone = op_constructor<VarPtr, Var*>("clone");
static auto make_detached_clone = op_constructor<VarPtr, Var*, bool>("clone");

CloneOp::CloneOp(Var* x) : CloneOp(x, false) {}

CloneOp::CloneOp(Var* x, bool detached) : x(x) {
    if (detached) set_stop_grad();
    set_flag(OpFlags::_cpu);
    set_flag(OpFlags::_cuda);
    set_flag(OpFlags::_manual_set_vnbb);
    y = create_output(nullptr, x->dtype());
    if (x->name.ptr)
        y->name = x->name;
}

VarPtr CloneOp::grad(Var* out, Var* dout, Var* v, int v_index) {
    return make_clone(dout);
}

void CloneOp::infer_shape() {
    y->set_shape(x->shape);
    y->storage_strides = x->storage_strides;
    y->share_with(x);
}

VarPtr detach(Var* x) {
    // A stopped producer does not observe or reconnect its input's history.
    // Mark it before Op::init, while ordinary clone keeps its recorded path.
    auto y = make_detached_clone(x, true);
    if (!x->autograd_alias) {
        x->autograd_alias = std::make_shared<AutogradAliasState>();
        x->autograd_alias->own_requires_grad = !x->is_stop_grad()
            && !x->flag(VarFlags::_requires_grad_disabled);
    }
    // Detach shares physical storage and write versions, but neither the
    // source's public trainability nor its gradient edge.
    y->autograd_alias = std::make_shared<AutogradAliasState>();
    y->autograd_alias->version = x->autograd_alias->version;
    y->autograd_alias->own_requires_grad = !y->is_stop_grad()
        && !y->flag(VarFlags::_requires_grad_disabled);
    return y;
}

VarPtr clone(Var* x) {
    return make_clone(x);
}

} // jittor
