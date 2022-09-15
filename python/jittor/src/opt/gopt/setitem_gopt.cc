// ***************************************************************
// Copyright (c) 2022 Jittor. All Rights Reserved.
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
    // return if output is all ready shared
    if (output->allocator) return;
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
    auto data_op = data->input();
    if (data_op->flags.get(NodeFlags::_custom_flag))
        return;

    auto in_shape = input->shape;
    int64 inplace_size = 1;
    for (int i = vs.n - 1; i > 0; --i) {
        VarSlice s = vs.slices[i];
        if (!(s.is_slice())) return;
        Slice ss = s.slice;
        if (!(ss.start == 0 && (ss.mask&2) && ss.step == 1))
            return;
        inplace_size *= in_shape[i];
    }
    
    VarSlice s = vs.slices[0];
    if (s.is_var() || s.is_str()) return;
    
    int64 size = 0;
    if (s.is_int())
        size = in_shape[0] == 0 ? 0 : s.i * input->size / in_shape[0];
    else if (s.is_slice()) {
        Slice ss = s.slice;
        // we also need to check the first dim is continuous
        if (ss.step != 1)
            return;
        size = in_shape[0] == 0 ? 0 : ss.start * input->size / in_shape[0];
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
    op->ns.set(GetitemOp::_inplace);
    // LOGir << input->shape << input->dtype() << data->shape << data->dtype() << vs << data->input();
    // LOGir << output;
}

static void getitem_inplace(GetitemOp* op) {
    // LOGir << "in getitem_inplace";

    auto in = op->inputs().front();
    auto ou = op->outputs().front();

    // return if out is all ready inplaced
    if (ou->allocator)
        return;

    VarSlices vs = op->vs;
    auto in_shape = in->shape;

    for (int i = vs.n - 1; i > 0; --i) {
        VarSlice s = vs.slices[i];
        if (!(s.is_slice())) return;
        Slice ss = s.slice;
        if (!(ss.start == 0 && (ss.mask&2) && ss.step == 1))
            return;
    }
    
    VarSlice s = vs.slices[0];
    if (s.is_var() || s.is_str()) return;
    
    int64 size = 0;
    if (s.is_int())
        size = in_shape[0] == 0 ? 0 : s.i * in->size / in_shape[0];
    else if (s.is_slice()) {
        size = in_shape[0] == 0 ? 0 : s.slice.start * in->size / in_shape[0];
        if (s.slice.step != 1) return;
    }
    ou->share_with(in, size);
    op->ns.set(GetitemOp::_inplace);
    // LOGir << "pass getitem_inplace";
    // LOGir << "inplace getitem" << vs << in->shape << ou->shape;
}

void SetitemOp::graph_optimize() {
    // LOGir << "hello graph_optimize";
    setitem_inplace(this);
    (void*)setitem_inplace;
}

void GetitemOp::graph_optimize() {
    // This optimize is still WIP
    // LOGir << "hello getitem graph_optimize";
    // setitem_grad_opt(this);
    // (void)getitem_inplace;
    getitem_inplace(this);
    (void*)getitem_inplace;
}

}

