// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <sstream>
#include "var.h"
#include "opt/pass_manager.h"
#include "opt/pass/insert_profile_loop_pass.h"

namespace jittor {

void InsertProfileLoopPass::run() {
    if (!op->get_loop_option("insert_profile_loop")) return;
    int loopend = ir->children.size()-1;
    auto check_loop = [](unique_ptr<KernelIR>& c) -> bool {
        return c->type == "loop" || c->has_attr("loop_func");
    };
    while (loopend>=0 && !check_loop(ir->children[loopend]))
        loopend--;
    if (loopend<0) {
        LOGw << "Loop body not found, profile loop cannot insert.";
        return;
    }
    int loopid = loopend;
    while (loopid>0 && check_loop(ir->children[loopid-1]))
        loopid--;
    vector<unique_ptr<KernelIR>> loops(loopend-loopid+1);
    for (int i=loopend, j=loops.size()-1; i>=loopid; i--, j--)
        loops[j] = ir->children[i]->move_out();
    
    ir->insert(loopid, "for (int _=0; _<1024; _++) {}");
    auto& loop = ir->children[loopid];
    loop->push_back("__asm__ __volatile__ (\"\": : : \"memory\");");
    loop->insert(1, loops);
}

} // jittor