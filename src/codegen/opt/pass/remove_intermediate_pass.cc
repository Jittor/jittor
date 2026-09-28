// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <sstream>
#include "core/var.h"
#include "codegen/opt/pass_manager.h"
#include "codegen/opt/pass/remove_intermediate_pass.h"

namespace jittor {

static bool remove_empty_loop(KernelIR* i) {
    for (int j=0; j<i->children.size(); j++) {
        if (remove_empty_loop(i->children[j].get()))
            j--;
    }
    if (i->type == KernelIRType::loop && i->children.size() == 0) {
        i->erase();
        return true;
    }
    return false;
}

// A kernel that reads back an output it stores declares no pointer
// `__restrict__`. With them, nvcc folds `c ? zp[i] : xp[i]` into one load
// from the selected address, issues it read-only and ahead of the store to
// `zp[i]` -- the read returns what the memory held before the kernel. Dropping
// the qualifier from `zp` alone is not enough; the select still carries `xp`'s.
static void unrestrict_if_reading_back(FusedOp* fused, KernelIR* ir) {
    unordered_set<Var*> read;
    for (Op* op : fused->ops)
        for (Var* v : op->inputs())
            read.insert(v);
    bool reads_back = false;
    for (auto& vi : fused->vars)
        reads_back |= vi.type == 2 && read.count(vi.var);
    if (!reads_back) return;
    ir->dfs([&](unique_ptr<KernelIR>& c) {
        if (c->type != KernelIRType::define) return;
        auto& dtype = c->get_attr(kir::dtype);
        auto at = dtype.find("__restrict__");
        if (at != string::npos) dtype.erase(at, sizeof("__restrict__")-1);
    });
}

void RemoveIntermediatePass::run() {
    unrestrict_if_reading_back(op, ir);
    unordered_set<string> names;
    for (auto& vi : op->vars) {
        // intermediate
        if (vi.type != 1) continue;
        Op* op = vi.var->input();
        if (!pm->oc->op_exist(op)) continue;
        for (uint i=0; i<op->outputs().size(); i++)
            if (op->output(i)==vi.var)
                names.insert(pm->oc->get_name_by_op_output(op, i));
    }
    LOGvvvv << "Remove intermediate:" << names;
    ir->remove_intermediate(names);
    ir->remove_all_unused();
    ir->solve_conflict_define();

    // remove empty loop
    remove_empty_loop(ir);

    ir->remove_all_unused();

}

} // jittor