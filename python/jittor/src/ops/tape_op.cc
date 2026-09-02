// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: 
//     Dun Liang <randonlang@gmail.com>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "var.h"
#include "ops/array_op.h"
#include "ops/op_register.h"
#include "ops/tape_op.h"

namespace jittor {

TapeOp::TapeOp(Var* x) {
    set_flag(OpFlags::_cpu);
    set_flag(OpFlags::_cuda);
    set_flag(OpFlags::_manual_set_vnbb);
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
    set_flag(OpFlags::_cpu);
    set_flag(OpFlags::_cuda);
    set_flag(OpFlags::_grads);
    set_flag(OpFlags::_manual_set_vnbb);
    callback = move(grad_callback);
    

    /*
                    stop grad        stop grad
        i --> tape --> t_i ---> .... ---> o --> tape --> t_o
        |                                         ^
        +---> tapes ------------------------------+
    */
    // set tape output
    for (int i=0; i<taped_outputs.size(); i++) {
        VarPtr out(0, taped_outputs[i]->var->dtype());
        out->add_inputs({this});
        auto v = taped_outputs[i]->var;
        auto op = v->input();
        // Wiring a new input into a tape that already ran corrupts its
        // liveness bookkeeping; fail loudly instead.
        ASSERT(op && !op->is_finished())
            << "tape output" << i << "must still be pending when taped together";
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
