// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "pybind/py_var_tracer.h"
#include "grad.h"
#include "var.h"
#include "op.h"
#include "graph.h"
#include "ops/op_register.h"

namespace jittor {

#define PREVENT_LARGE_FUSED_OP 16

static auto make_binary = get_op_info("binary")
    .get_constructor<VarPtr, Var*, Var*, NanoString>();
static auto make_number = get_op_info("number")
    .get_constructor<VarPtr, float, Var*>();


VarPtr make_grad(Op* op, Var* out, Var* dout, Var* x, int x_index) {
    if (dout == nullptr) return nullptr;
    LOGvvvv << "Make grad op:" >> op->name() << "inputs:" >> op->inputs()
        << "out:" >> out << "dout:" >> dout << "x:" >> x << "xid:" >> x_index;
    return op->grad(out, dout, x, x_index);
}

inline static void assign_attrs(Var* a, Var* b) {
    if (b->flags.get(NodeFlags::_stop_fuse))
        a->flags.set(NodeFlags::_stop_fuse);
}

vector<VarPtr> grad(Var* loss, vector<Var*> targets) {
    LOGvv << "loss:" >> loss << "targets:" >> targets;
    CHECK(loss->is_float()) << "Loss should be float";
    for (Var* var : targets)
        CHECK(var->is_float()) << "Targets of grad should be float";
    // successors of targets
    vector<Node*> ts(targets.begin(), targets.end());
    // bfs visit find all successors of targets
    LOGvv << "Size of successors:" << ts.size();
    bfs_forward(ts, [](Node*){ return true; });
    vector<Node*> gnodes;
    gnodes.reserve(ts.size());
    auto nt = Node::tflag_count;
    if (loss->tflag == nt)
        gnodes.push_back(loss);
    bfs_backward(gnodes, [&](Node* node) {
        if (node->tflag != nt)
            return false;
        if (node->is_stop_grad())
            return false;
        // int value has zero grad
        if (node->is_var())
            return node->var()->is_float();
        return true;
    });
    LOGvv << "Size of grad nodes:" << gnodes.size();
    
    vector<Node*> sorted;
    toplogical_sort_backward(gnodes, sorted, [](Node*){});
    nt = Node::tflag_count;
    vector<Var*> gvars;
    gvars.reserve(sorted.size());
    for (Node* node : sorted)
        if (node->is_var())
            gvars.push_back(node->var());
    LOGvv << "Size of grad vars:" << gvars.size();
    
    vector<VarPtr> grads(gvars.size());
    vector<VarPtr> results(targets.size());
    for (size_t i=0; i<gvars.size(); i++)
        gvars[i]->custom_data = i;
        
    for (size_t i=0; i<gvars.size(); i++) {
        Var* var = gvars[i];
        auto& grad = grads[i];
        #ifdef PREVENT_LARGE_FUSED_OP
        int gsum = 0;
        #endif
        if (i==0) {
            grad = make_number(1.f, loss);
            assign_attrs(grad.ptr, loss);
            registe_node_trace_grad(grad.ptr, loss, 0);
        } else
        for (auto it : var->outputs_with_index()) {
            Op* op = it.op;
            auto index = it.index;
            if (op->tflag != nt) continue;
            // TODO: support two outputs backprop.
            Var* out = op->outputs().back();
            Var* dout = grads[out->custom_data];
            VarPtr dvar = make_grad(op, out, dout, var, index);
            registe_node_trace_grad(dvar.ptr, op, index);
            if (dvar)
                ASSERT(dvar->num==var->num && dvar->shape.size()==var->shape.size())
                << "dvar" << dvar << "var" << var;
            if (!grad)
                grad = move(dvar);
            else if (dvar) {
                grad = make_binary(grad, dvar, ns_add);
                #ifdef PREVENT_LARGE_FUSED_OP
                gsum ++;
                if (gsum>=PREVENT_LARGE_FUSED_OP) {
                    // TODO: this is a dirty fix for
                    // stopping fuse lots of op together,
                    // try to find a better solution
                    grad->flags.set(NodeFlags::_stop_fuse);
                }
                #endif
                assign_attrs(grad.ptr, var);
                registe_node_trace_grad(grad.ptr, var, index);
            }
        }
    }
    // set zero grad
    for (size_t i=0; i<results.size(); i++) {
        Var* var = targets[i];
        VarPtr& grad = results[i];
        if (var->tflag == nt)
            grad = move(grads[var->custom_data]);
        if (!grad) {
            LOGw << "grads[">>i>>"] doesn't have gradient. It will be set to zero:" << var;
            grad = make_number(0.f, var);
            assign_attrs(grad.ptr, var);
            registe_node_trace_grad(grad.ptr, var, 0);
        }
    }
    return results;
}

} // jittor