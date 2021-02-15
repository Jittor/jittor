// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <sstream>
#include "var.h"
#include "op_compiler.h"
#include "opt/pass_manager.h"
#include "opt/pass/compile_shapes_pass.h"

namespace jittor {

void CompileShapesPass::run() {
    if (!op->get_loop_option("compile_shapes")) return;
    for (auto& c : ir->children) {
        if (c->type != "define") continue;
        auto& rvalue = c->get_attr("rvalue");
        // T range = op{i}_{vnamr}->shape[j];
        //                        j      i
        if (!startswith(rvalue, "op") || rvalue.back() != ']')
            continue;
        uint i=rvalue.size()-2;
        while (i && isdigit(rvalue[i])) i--;
        ASSERT(rvalue[i] == '[' && i>7);
        uint j = i-7;
        ASSERT(startswith(rvalue, "->shape[", j));
        string name = rvalue.substr(0, j);
        uint op_id, opvar_id;
        Op* op;
        Var* var;
        pm->oc->get_op_var_by_name(name, op_id, opvar_id, op, var);
        int shapeid = std::stoi(rvalue.substr(i+1, rvalue.size()-i-2));
        ASSERT(shapeid < (int)var->shape.size());
        rvalue = S(var->shape[shapeid]);
    }
}

} // jittor