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
#include "ops/op_register.h"

namespace jittor {

inline static bool fast_strcmp(const char* a, const char* b) {
    while (*b && *a == *b) a++, b++;
    return !*b;
}

static auto make_empty = get_op_info("empty")
    .get_constructor<VarPtr, NanoVector, NanoString>();

static void setitem_inplace(SetitemOp* op) {
    LOGvvvv << "setitem_inplace";
    if (!op->flags.get(NodeFlags::_has_gopt))
        return;
    auto input = op->inputs().front(); 
    if (!(input->backward_liveness<=1 &&
        (op->op == ns_void || op->op == ns_add || op->op == ns_subtract))) {
        return;
    }
    auto input_op = input->input();
    if (input_op) {
        // make sure input op will not use input
        auto input_name = input_op->name();
        // if it is not setitem and been inplaced
        if (!fast_strcmp(input_name, "setitem") &&
            (!input->mem_ptr && input->allocator))
        return;
    }

    auto output = op->outputs().front();
    output->share_with(input);
    
    // data shares memory with input
    auto data = op->input(1);
    input_op = input->input();

    if (input_op && input_op->inputs().size() == 1) {
        input_op = input_op->inputs().front()->input();
    }
    if (input_op && input_op->inputs().size() == 1) {
        input_op = input_op->inputs().front()->input();
    }

    VarSlices vs = op->vs;
    /* data can share memory with input, which must suit:
        - data must be not finished
        - data has no other output
    */
    if (data->is_finished() || data->backward_liveness>1)
    //  || input_op || input_op->inputs().size() == 0)))
        return;

    auto in_shape = input->shape;
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
        size = s.i * input->size / in_shape[0];
    else if (s.is_slice())
        size = s.slice.start * input->size / in_shape[0];
    
    data->input()->add_inputs(vector<Var*>{input});
    data->share_with(input, size);
    op->flags.set(NodeFlags::_has_gopt, 0);
    LOGvvvv << "setitem_inplace happens";
}

static void getitem_grad_opt(SetitemOp* op) {
    LOGvvvv << "getitem_grad_opt";
    if (!op->flags.get(NodeFlags::_has_gopt))
        return;

    bool last = true;
    SetitemOp* last_set_op = nullptr;
    Var* last_dv = nullptr;
    while (1) {
        // op is a setitem op, out is dv
        auto cur_dv = op->outputs().front();
        setitem_inplace(op);

        // out_op is a binary.add op
        auto dv_out_op = cur_dv->outputs().front();

        if (dv_out_op == nullptr) return;
        if (dv_out_op && !fast_strcmp(dv_out_op->name(), "binary")) return;

        Var* pre_dv = nullptr;
        for (auto* tmp : dv_out_op->inputs()) {
            if (tmp != cur_dv) { pre_dv = tmp; break; }
        }

        auto pre_dv_in_op = pre_dv->input();

        if (last) {
            last_dv = dv_out_op->outputs().front();
            last_set_op = op;
            last = false;
        }

        if (fast_strcmp(pre_dv_in_op->name(), "binary")) {
            for (auto* tmp : pre_dv_in_op->inputs()) {
                if (fast_strcmp(tmp->input()->name(), "setitem")) {
                    pre_dv = tmp;
                    break;
                }
            }
            op->set_inputs(list<Node*>{pre_dv, op->inputs().back()});
            op = (SetitemOp *)(pre_dv->input());
            op->flags.set(NodeFlags::_has_gopt, 0);
        }
        else if (fast_strcmp(pre_dv_in_op->name(), "setitem")) {
            op->set_inputs(list<Node*>{pre_dv, op->inputs().back()});

            auto ori_v = pre_dv->input()->inputs().front();
            auto tmp_v = make_empty(ori_v->shape, ori_v->dtype());
            ori_v->set_inputs({{tmp_v->input()}});
            tmp_v->set_inputs({});
            op->flags.set(NodeFlags::_has_gopt, 0);
            pre_dv->input()->flags.set(NodeFlags::_has_gopt, 0);
            break;
        }
        
    }
    last_dv->set_inputs({{last_set_op}});
    if (last_set_op->outputs().size() == 2) {
        last_set_op->outputs().front()->set_inputs({});
        ASSERT(last_set_op->outputs().size() == 1) << last_set_op->outputs();
    }
    LOGvvvv << "getitem_grad_opt happens";
}

static void setitem_grad_opt(GetitemOp* op) {
    LOGvvvv << "setitem_grad_opt";
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
    if (chain.size() == 0) return;
    SetitemOp* cur_op = chain[0];
    VarSlices vs = chain[0]->vs;

    // only suppot :*n, int or slice now
    int idx_min = -1;
    int idx_max = -1;
    int idx = -1;
    for (int i = 0; i < vs.n; ++i) {
        VarSlice s = vs.slices[i];
        if (s.is_int()) {
            idx_min = s.i;
            idx_max = s.i;
            if (idx != -1) return;
            idx = i;
        }
        else if (s.is_slice()) {
            idx_min = s.slice.start;
            idx_max = s.slice.stop;
            if (idx_min == 0 && idx_max == -1) continue;
            if (idx != -1) return;
            idx = i;
        }
    }
    
    for (auto* sop : chain) {
        auto out_var = sop->outputs().front();
        auto in_var = cur_op->inputs().front();
        for (auto* out : out_var->outputs()) {
            if (fast_strcmp(out->name(), "getitem")) {
                GetitemOp* cur_get_op = (GetitemOp*)out;
                VarSlices vs = cur_get_op->vs;

                int cur_idx_min = -1, cur_idx_max = -1;
                for (int i = 0; i < vs.n; ++i) {
                    VarSlice s = vs.slices[i];
                    if (s.is_int()) {
                        cur_idx_min = s.i;
                        cur_idx_max = s.i;
                        if (i != idx) return;
                    }
                    else if (s.is_slice()) {
                        cur_idx_min = s.slice.start;
                        cur_idx_max = s.slice.stop-1;
                        if (cur_idx_min == 0 && cur_idx_max == -2) continue;
                        if (i != idx) return;
                    }
                }

                int flag = 0;
                // 括号数组，如果当前的区间与之前记录的区间没有overlap就直接share memory
                if (cur_idx_max < idx_min) {
                    idx_min = cur_idx_min;
                    flag = 1;
                }
                else if (cur_idx_min > idx_max) {
                    idx_max = cur_idx_max;
                    flag = 1;
                }
                else
                    cur_op = sop;
                
                if (flag == 1) {
                    LOGvvvv << "setitem_grad_opt set success";
                    cur_get_op->set_inputs({in_var});
                }
            }
            out->flags.set(NodeFlags::_has_gopt, 0);
        }
    }
    LOGvvvv << "setitem_grad_opt happens";
}

static void getitem_inplace(GetitemOp* op) {
    LOGvvvv << "getitem_inplace";

    if (!op->flags.get(NodeFlags::_has_gopt))
        return;

    auto in = op->inputs().front();
    auto ou = op->outputs().front();
    
    // return if input or output's shape is variable
    if (in->num < 0 || ou->num < 0)
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
    op->flags.set(NodeFlags::_has_gopt, 0);
    LOGvvvv << "getitem_inplace happens";
}

void SetitemOp::graph_optimize() {
    // LOGvvvv << "hello graph_optimize";
    // (void)getitem_grad_opt;
    getitem_grad_opt(this);
    setitem_inplace(this);
}

void GetitemOp::graph_optimize() {
    // This optimize is still WIP
    // LOGvvvv << "hello getitem graph_optimize";
    setitem_grad_opt(this);
    getitem_inplace(this);
}

}

