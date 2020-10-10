// ***************************************************************
// Copyright (c) 2020 Jittor. All Rights Reserved.
// Authors: Dun Liang <randonlang@gmail.com>. 
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

static void setitem_inplace(SetitemOp* op) {
    // LOGir << "setitem_inplace";
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
            fast_strcmp(input_name, "array") || 
            fast_strcmp(input_name, "empty") || 
            fast_strcmp(input_name, "setitem") ||
            fast_strcmp(input_name, "getitem")))
            // TODO: inplace getitem maybe risky, getitem maybe inplace too
        return;
    }
    auto output = op->outputs().front();
    output->share_with(input);
    // LOGir << "apply setitem_inplace on" << op << "input:" << input << "output:" << output;
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

void SetitemOp::graph_optimize() {
    // LOGir << "hello graph_optimize";
    setitem_inplace(this);
}

void GetitemOp::graph_optimize() {
    // This optimize is still WIP
    // LOGir << "hello getitem graph_optimize";
    // setitem_grad_opt(this);
    (void)setitem_grad_opt;
}

}

