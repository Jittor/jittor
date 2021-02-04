// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved.
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <cmath>
#include "var.h"
#include "ops/setitem_op.h"
#include "ops/getitem_op.h"

namespace jittor {

inline static bool fast_strcmp(const char* a, const char* b) {
    while (*b && *a == *b) a++, b++;
    return !*b;
}

// add dependency b -> a
static inline void add_dependency(Node* a, const vector<Node*>& b) {
    a->add_inputs(b);
    auto edge = a->_inputs.end();
    for (int i=0; i<b.size(); i++) {
        edge = std::prev(edge);
        // set -1 mean this is a control dependency edge
        edge->back->index = -1;
    }
}

static void setitem_inplace(SetitemOp* op) {
    // LOGir << "in setitem_inplace";
    auto input = op->inputs().front();
    if (!(input->outputs().size() == 1 && 
        input->forward_liveness<=1 &&
        (op->op == ns_void || op->op == ns_add || op->op == ns_subtract))) {
        return;
    }
    auto input_op = input->input();
    if (input_op) {
        // make sure input op will not use input
        auto input_name = input_op->name();
        if (!(input_op->type() == OpType::broadcast || 
            input_op->inputs().size() == 0 ||
            fast_strcmp(input_name, "setitem") ||
            fast_strcmp(input_name, "getitem")))
            // TODO: inplace getitem maybe risky, getitem maybe inplace too
        return;
    }
    auto output = op->outputs().front();
    output->share_with(input);
    
    auto data = op->input(1);
    // if setitem requires type conversion, don't inplace
    if (data->dtype() != input->dtype())
        return;

    input_op = input->input();

    if (input_op && input_op->inputs().size() == 1) {
        input_op = input_op->inputs().front()->input();
    }
    if (input_op && input_op->inputs().size() == 1) {
        input_op = input_op->inputs().front()->input();
    }

    VarSlices vs = op->vs;
    if (!(data->is_finished() == 0 && 
          (data->outputs().size() == 1 || 
           (!input_op 
            || input_op->inputs().size() == 0))))
        return;
    if (data->allocator)
        return;

    auto in_shape = input->shape;
    int64 inplace_size = 1;
    for (int i = vs.n - 1; i > 0; --i) {
        VarSlice s = vs.slices[i];
        if (!(s.is_slice())) return;
        Slice ss = s.slice;
        if (!(ss.start == 0 && ss.stop >= in_shape[i] && ss.step == 1))
            return;
        inplace_size *= in_shape[i];
    }
    
    VarSlice s = vs.slices[0];
    if (s.is_var()) return;
    
    auto size = 0;
    if (s.is_int())
        size = s.i * input->size / in_shape[0];
    else if (s.is_slice()) {
        Slice ss = s.slice;
        // we also need to check the first dim is continuous
        if (ss.step != 1)
            return;
        size = ss.start * input->size / in_shape[0];
        inplace_size *= ss.stop - ss.start;
    }
    if (inplace_size > data->num) {
        // if data has been broadcast into input, don't
        // inplace data, because their shapes are not match
        // This would lead partial setitem
        return;
    }
    add_dependency(data->input(), {input->node()});
    data->share_with(input, size);
}

struct BBox {
    int n = 0;
    int* minmax = nullptr;
    


    void load_var_slice(const VarSlice& vs) {

    }
};

static void setitem_grad_opt(GetitemOp* op) {
    if (!op->flags.get(NodeFlags::_has_gopt))
        return;
    auto get_in = op->inputs().front();
    auto get_in_op = get_in->input();
    if (!get_in_op)
        return;
    auto name = get_in_op->name();
    if (!fast_strcmp(name, "setitem"))
        return;
    // find setitem op chain
    auto first_set = (SetitemOp*)get_in_op;
    vector<SetitemOp*> chain;
    while (1) {
        auto next = first_set->inputs().front()->input();
        if (!next) break;
        if (!fast_strcmp(next->name(), "setitem"))
            break;
        chain.push_back(first_set);
        first_set = (SetitemOp*)next;
    }
    chain.push_back(first_set);
    for (int i=0; i<chain.size()/2; i++)
        std::swap(chain[i], chain[chain.size()-1-i]);
    auto last_set = (SetitemOp*)get_in_op;
    while (1) {
        SetitemOp* next = nullptr;
        auto out_var = last_set->outputs().front();
        for (auto* out : out_var->outputs()) {
            if (fast_strcmp(out->name(), "setitem")) {
                next = (SetitemOp*)out;
                break;
            }
        }
        if (!next) break;
        last_set = next;
        chain.push_back(next);
    }
    // LOGir << "find setitem chain" << chain.size() << chain;
    for (auto* sop : chain) {
        // LOGig << sop << sop->vs;
        auto out_var = sop->outputs().front();
        for (auto* out : out_var->outputs()) {
            if (fast_strcmp(out->name(), "getitem")) {
                out->flags.set(NodeFlags::_has_gopt, 0);
            }
        }
    }

}

static void getitem_inplace(GetitemOp* op) {
    // LOGir << "in getitem_inplace";

    auto in = op->inputs().front();
    auto ou = op->outputs().front();
    
    // return if input or output's shape is variable
    if (in->num <= 0 || ou->num <= 0)
        return;

    VarSlices vs = op->vs;
    auto in_shape = in->shape;

    for (int i = vs.n - 1; i > 0; --i) {
        VarSlice s = vs.slices[i];
        if (!(s.is_slice())) return;
        Slice ss = s.slice;
        if (!(ss.start == 0 && ss.stop >= in_shape[i] && ss.step == 1))
            return; 
    }
    
    VarSlice s = vs.slices[0];
    if (s.is_var()) return;
    
    auto size = 0;
    if (s.is_int())
        size = s.i * in->size / in_shape[0];
    else if (s.is_slice())
        size = s.slice.start * in->size / in_shape[0];
    ou->share_with(in, size);
    // LOGir << "pass getitem_inplace";
}

void SetitemOp::graph_optimize() {
    // LOGir << "hello graph_optimize";
    setitem_inplace(this);
    (void)setitem_inplace;
}

void GetitemOp::graph_optimize() {
    // This optimize is still WIP
    // LOGir << "hello getitem graph_optimize";
    setitem_grad_opt(this);
    (void)setitem_grad_opt;
    // (void)getitem_inplace;
    getitem_inplace(this);
    (void)getitem_inplace;
}

}

