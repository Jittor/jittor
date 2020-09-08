// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <sstream>
#include "var.h"
#include "op_compiler.h"
#include "mem/allocator.h"
#include "opt/pass_manager.h"
#include "opt/pass/assume_aligned_pass.h"
#include "executor.h"

namespace jittor {

void AssumeAlignedPass::run() {
    if (!op->get_loop_option("compile_shapes")) return;
    ir->push_front("#define assume_aligned(ptr) (void)(__builtin_assume_aligned(ptr, alignment))", &ir->before);
    auto check = [&](KernelIR* func) {
        if (func->type != "func")
            return;
        vector<unique_ptr<KernelIR>>* ls[] = {&func->inner, &func->children};
        for (auto& l : ls)
            for (auto& c : (*l)) {
                if (c->type != "define") continue;
                auto& lvalue = c->get_attr("lvalue");
                // if is a var pointer
                if (startswith(lvalue, "op") && endswith(lvalue, "p")) {
                    string name = lvalue.substr(0, lvalue.size()-1);
                    uint op_id, opvar_id;
                    Op* op;
                    Var* var;
                    pm->oc->get_op_var_by_name(name, op_id, opvar_id, op, var);
                    // add assume_aligned if is aligned_allocator
                    if (exe.allocator->is_aligned()) {
                        // if is a function arguments
                        if (l == ls[0])
                            func->push_front("assume_aligned("+lvalue+");");
                        else
                            c->push_back("assume_aligned("+lvalue+");", &c->after);
                    }
                }
            }

    };
    check(ir);
    for (auto& c : ir->before)
        check(c.get());
}

} // jittor