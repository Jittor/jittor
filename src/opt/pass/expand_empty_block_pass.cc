// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <sstream>
#include "var.h"
#include "opt/pass_manager.h"
#include "opt/pass/expand_empty_block_pass.h"

namespace jittor {

void check_empty_block(KernelIR* ir) {
    for (uint i=0; i<ir->children.size(); i++) {
        auto loop = ir->children[i].get();
        if (loop->type != "loop") continue;
        if (loop->has_attr("loop_id")) {
            continue;
        }
        if (loop->has_attr("rvalue"))
            continue;
        ir->insert(i+1, "for (int _=0; _<1; _++) {}");
        ir->children[i+1]->insert(0, loop->children);
        // use children[i] instead of loop
        ir->children[i]->erase();
        i--;
    }
}

void ExpandEmptyBlockPass::run() {
    check_empty_block(ir);
    ir->expand_empty_block();
}

JIT_TEST(check_empty_block) {
    KernelIR ir("x=1;{a=1;}y=1;");
    check_empty_block(&ir);
    ASSERT(ir.children[1]->attrs.at("lvalue")=="_");
    ir.move_loop_back();
    ASSERT(ir.children[2]->attrs.at("lvalue")=="_");
}

} // jittor