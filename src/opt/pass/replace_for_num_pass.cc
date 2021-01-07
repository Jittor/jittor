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
#include "opt/pass/replace_for_num_pass.h"

namespace jittor {

void ReplaceForNumPass::run() {
    for (uint fid=0; fid<ir->children.size(); fid++) {
        auto& loop_ir = ir->children[fid];
        if (loop_ir->type != "loop")
            continue;
        auto& rvalue = loop_ir->get_attr("rvalue");
        auto j=rvalue.find("num");
        if (j == string::npos) continue;
        auto& loop_num = rvalue;
        auto& loop_index = loop_ir->get_attr("lvalue");
        LOGvvvv << "Find for_num" << loop_num << loop_index;
        uint sid=fid-1;
        bool found = false;
        // find definition of loop range
        for (;sid>0; sid--) {
            if (ir->children[sid]->type != "define")
                continue;
            if (!ir->children[sid]->check_attr("lvalue", loop_num))
                continue;
            found = true;
            break;
        }
        // T xx_num = xxx->num
        // def = xxx
        ASSERT(found);
        auto& code2 = ir->children[sid]->get_attr("rvalue");
        ASSERT(endswith(code2, "->num")) << ir->children[sid]->attrs;
        string def = code2.substr(0, code2.size()-5);
        uint op_id, opvar_id;
        Op* op;
        Var* var;
        pm->oc->get_op_var_by_name(def, op_id, opvar_id, op, var);
        auto new_code = OpCompiler::precompile(
            {
                {"DIM", S(var->shape.size())},
                {"op_id", S(op_id)},
                {"def", def},
                {"loop_index", loop_index},
            } ,
                "@for(di,0,DIM, op@op_id@@_index_t @def@@shape@di = @def->shape[@di];)\n"
                "op@op_id@@_index_t @def@@stride@{DIM-1} = 1;\n"
                "@for(di,DIM-2,-1,-1, auto @def@@stride@di = @def@@stride@{di+1} * @def@@shape@{di+1};)\n"
                "@for(di,0,DIM, for (op@op_id@@_index_t @loop_index@di=0; @loop_index@di<@def@@shape@di; @loop_index@di++))\n"
                "{ op@op_id@@_index_t @loop_index = @for(di,0,DIM, + @loop_index@di * @def@@stride@di); }"
        );
        KernelIR new_ir(new_code);
        ASSERT(new_ir.children.size()>=2 &&
            new_ir.children.back()->type == "loop" &&
            new_ir.children.front()->type == "define");
        auto& new_for = new_ir.children.back();
        auto* inner_for = new_for.get();
        for (uint di=0; di+1<var->shape.size(); di++) {
            ASSERT(inner_for->children.size()==1);
            inner_for = inner_for->children[0].get();
        }
        auto& prev_for = ir->children[fid];
        LOGvvvv << "new_ir\n" >> new_ir.to_string();
        LOGvvvv << "prev_for\n" >> prev_for->to_string();
        inner_for->insert(
            inner_for->children.size(),
            prev_for->children
        );
        LOGvvvv << "new_ir\n" >> new_ir.to_string();
        prev_for->erase();
        fid += new_ir.children.size()-1;
        ir->insert(sid, new_ir.children);
    }
}

} // jittor