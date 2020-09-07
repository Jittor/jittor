// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <sstream>
#include "var.h"
#include "opt/pass_manager.h"
#include "opt/pass/loop_var_analyze_pass.h"
#include "ops/reduce_op.h"
#include "ops/broadcast_to_op.h"

namespace jittor {

void LoopVarAnalyzePass::run() {
    // loop_vars: opi_xx->shape[j]
    vector<string> loop_vars;
    // we use input var of reduce op as the loop var
    // TODO: consider reshape op
    auto& vars = op->vars;
    bool has_reduce = false, has_element = false;
    bool has_op = false;
    for (Op* op : this->op->ops) {
        auto& op_members = this->pm->oc->op_members;
        // TODO: fix it
        // ugly temp fix for index_var
        auto opid = this->op->get_node_id(op);
        if (op->name()==string("index") && 
            op->inputs().size()+op->outputs().size() != op_members[opid].size()) {
            op_members[opid].insert(op_members[opid].begin(), "wtf");
        }
    }
    // LoopVarAnalyzePass has three steps:
    // 1. Find the appropriate variable and use its shape as loop variables.
    //      Those loop vars are store in "vector<string> loop_vars",
    //      e.g. op0_x->shape[0], op0_x->shape[1], op0_x->shape[2], ...
    // 2. Replace the loop variable of the fused op with those loop variable.
    //
    //      For example, previous code is :
    // 
    //      index_t op0_xshape0 = op0_x->shape[0];
    //      index_t op0_xshape1 = op0_x->shape[1];
    //      index_t op0_xshape2 = op0_x->shape[2];
    //      for (index_t op0_i0=0; op0_i0<op0_xshape0; op0_i0++)
    //          for (index_t op0_i1=0; op0_i1<op0_xshape1; op0_i1++)
    //              for (index_t op0_i2=0; op0_i2<op0_xshape2; op0_i2++)
    //                  ......
    //      index_t op1_yshape0 = op1_y->shape[0];
    //      index_t op1_yshape1 = op1_y->shape[1];
    //      index_t op1_yshape2 = op1_y->shape[2];
    //      for (index_t op1_y0=0; op1_y0<op1_yshape0; op1_y0++)
    //          for (index_t op1_y1=0; op1_y1<op1_yshape1; op1_y1++)
    //              for (index_t op1_y2=0; op1_y2<op1_yshape2; op1_y2++)
    //                  ......
    //
    //      After replace:
    //      
    //      index_t range0 = op0_x->shape[0];
    //      index_t range1 = op0_x->shape[1];
    //      index_t range2 = op0_x->shape[2];
    //      for (index_t op0_i0=0; op0_i0<range0; op0_i0++)
    //          for (index_t op0_i1=0; op0_i1<range1; op0_i1++)
    //              for (index_t op0_i2=0; op0_i2<range2; op0_i2++)
    //                  ......
    //      for (index_t op1_y0=0; op1_y0<range0; op1_y0++)
    //          for (index_t op1_y1=0; op1_y1<range1; op1_y1++)
    //              for (index_t op1_y2=0; op1_y2<range2; op1_y2++)
    //                  ......
    //
    // 3. Change the different aliases of the same variable to the same name
    //      For example, consider a computing graph: 
    //      op1 --> op2, op2's input is an alias of op1's output,
    //      Suppose the input of op2 is op2_x, the output of op1 is op1_y
    //      we replace op2_x with op1_y

    // TODO: find loop range in better way
    // we pick loop var from below priority:
    // 1. reduce input
    // 2. element input
    // 3. broadcast output
    for (uint i=0; i<vars.size(); i++) {
        // output
        if (vars[i].type == 2) {
            Var* var = vars[i].var;
            Op* op = var->input();
            if (!pm->oc->op_exist(op))
                continue;
            has_op = true;
            if (op->type() == OpType::reduce)
                has_reduce = true;
            if (op->type() == OpType::element)
                has_element = true;
        }
    }
    for (uint i=0; i<vars.size(); i++) {
        // output
        if (vars[i].type == 2) {
            Var* var = vars[i].var;
            Op* op = var->input();
            // input var as loop var
            // TODO: consider only broadcast
            var = op->inputs().front();
            if (!pm->oc->op_exist(op))
                continue;
            if (has_reduce && op->type() != OpType::reduce)
                continue;
            if (has_element && !has_reduce && op->type() != OpType::element)
                continue;
            Var* loop_var;
            if (op->type() == OpType::broadcast || op->name_ex() == "index") {
                loop_var = op->output(0);
            } else {
                loop_var = op->inputs().front();
            }
            loop_vars.reserve(loop_var->shape.size());
            string vname = pm->oc->get_name_by_op_var(op, loop_var);
            for (uint j=0; j<loop_var->shape.size(); j++)
                loop_vars.emplace_back(vname+"->shape["+S(j)+"]");
            break;
        }
    }
    ASSERT(!has_op || loop_vars.size()) << "Loop var not found." << op->ops;
    // if (loop_vars.size()==0) {
    //     LOGw << "TODO: loop var not found.";
    //     // return;
    // }
    vector<unique_ptr<KernelIR>> loop_var_defines;
    vector<string> loop_var_names;
    vector<string> unused;
    for (uint k=0; k<loop_vars.size(); k++) {
        std::stringstream loop_var_define;
        auto opi = split(loop_vars[k], "_").at(0);
        // op{op_i}_index_t range{k} = {var_i_name}->shape[{j}]
        loop_var_define << opi << "_index_t range" << k << 
            " = " << loop_vars[k] << ";";
        loop_var_defines.emplace_back(
            std::make_unique<KernelIR>(loop_var_define.str()));
        loop_var_names.emplace_back(string("range")+S(k));
        unused.emplace_back(loop_var_names.back());
    }
    number_of_ranges = loop_var_names.size();
    int member_count=pm->oc->total_member_count();

    ir->insert(member_count, loop_var_defines);
    // replace loop var
    vector<pair<string,string>> replace_vars;
    for (uint i=0; i<op->ops.size(); i++) {
        Op* opi = op->ops[i];
        uint ndim=0;
        uint64_t mask=0;
        vector<string> vnames;
        // loop var may not exist(relayed)
        if (!pm->oc->op_exist(opi))
            continue;
        if (opi->name()==string("array"))
            continue;
        if (opi->type() == OpType::reduce) {
            ndim = ((ReduceOp*)opi)->inputs().front()->shape.size();
            for (uint i=0; i<opi->inputs().size(); i++)
                vnames.push_back(pm->oc->get_name_by_op_input(opi, i));
        } else 
        if (opi->type() == OpType::broadcast) {
            ndim = ((BroadcastToOp*)opi)->outputs().front()->shape.size();
            for (uint o=0; o<opi->outputs().size(); o++)
                vnames.push_back(pm->oc->get_name_by_op_output(opi, o));
        } else {
            ndim = opi->outputs().front()->shape.size();
            for (uint o=0; o<opi->outputs().size(); o++)
                vnames.push_back(pm->oc->get_name_by_op_output(opi, o));
        }
        for (uint j=0; j<ndim; j++)
            if (!(mask>>j&1) && j<loop_var_names.size()) {
                for (auto& vname : vnames) {
                    // cannot replace extras shape
                    // TODO: optimize it
                    if (vname.find("extras") != string::npos)
                        continue;
                    // replace op{i}_{vname}shape{j} -> {loop_var_names[j]}
                    std::stringstream name1;
                    name1 << vname<<"shape"<<j;
                    auto& name2 = loop_var_names[j];
                    replace_vars.emplace_back(name1.str(), name2);
                }
            }
    }
    
    for (auto& t : op->edges) {
        uint i,j,k,l;
        std::tie(i,j,k,l) = t;
        // loop var may not exist(relayed)
        auto opa = op->ops[i];
        auto opb = op->ops[k];
        if (!pm->oc->op_exist(opa) || !pm->oc->op_exist(opb))
            continue;
        // replace op{j}_{kname}* -> op{i}_{oname}*
        auto name1 = pm->oc->get_name_by_op_input(opb, l);
        auto name2 = pm->oc->get_name_by_op_output(opa, j);
        replace_vars.emplace_back(name1, name2);
    }
    
    LOGvvv << "replace_vars" << replace_vars;
    ir->replace(replace_vars);
    LOGvvvv << "KernelIR after replace\n" >> ir->to_string(0, true);
    // move define
    ir->move_loop_back();
    LOGvvvv << "KernelIR after move_loop_back\n" >> ir->to_string(0, true);
}

} // jittor