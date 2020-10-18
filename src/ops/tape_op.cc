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

TapeOp::TapeOp(Var* x) {
    flags.set(NodeFlags::_cpu);
    flags.set(NodeFlags::_cuda);
    create_output(nullptr, x->dtype());
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
    CHECK(callback.deleter);
    try {
        callback.func(_outputs.size(), douts, _inputs.size(), dins);
    } catch (...) {
        // if error occur in callback, we need to
        // free it to prevent memory leak, but this is still
        // not enough, error may occur outside. please
        // find a better solution
        callback.deleter();
        callback.deleter = nullptr;
        throw;
    }
}

Tapes::Tapes(
    const vector<VarHolder*>& taped_inputs,
    const vector<VarHolder*>& taped_outputs,
    GradCallback&& grad_callback
) {
    flags.set(NodeFlags::_cpu);
    flags.set(NodeFlags::_cuda);
    flags.set(NodeFlags::_grads);
    callback = move(grad_callback);
    

    /*
                    stop grad        stop grad
        i --> tape --> t_i ---> .... ---> o --> tape --> t_o
        |                                         ^
        +---> tapes ------------------------------+
    */
    // set tape output
    for (int i=0; i<taped_outputs.size(); i++) {
        VarPtr out(0, ns_float32);
        out->add_inputs({this});
        auto v = taped_outputs[i]->var;
        auto op = v->input();
        op->add_inputs(vector<Node*>{out.ptr});
    }
    // set tapes input 
    vector<Var*> tin(taped_inputs.size());
    for (int i=0; i<taped_inputs.size(); i++) {
        tin[i] = taped_inputs[i]->var->input()->inputs().front();
    }
    add_inputs(tin);
    // stop grad for input and output
    for (int i=0; i<taped_inputs.size(); i++) {
        taped_inputs[i]->var->set_stop_grad();
    }
    for (int i=0; i<taped_outputs.size(); i++) {
        taped_outputs[i]->var->input()->inputs().front()->set_stop_grad();
    }
}

void tape_together(
    const vector<VarHolder*>& taped_inputs,
    const vector<VarHolder*>& taped_outputs,
    GradCallback&& grad_callback
) {
    new Tapes(taped_inputs, taped_outputs, move(grad_callback));
}

} // jittor
