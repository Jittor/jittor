// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <sstream>
#include "var.h"
#include "opt/pass_manager.h"
#include "opt/pass/remove_intermediate_pass.h"

namespace jittor {

static bool remove_empty_loop(KernelIR* i) {
    for (int j=0; j<i->children.size(); j++) {
        if (remove_empty_loop(i->children[j].get()))
            j--;
    }
    if (i->type == "loop" && i->children.size() == 0) {
        i->erase();
        return true;
    }
    return false;
}

void RemoveIntermediatePass::run() {
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