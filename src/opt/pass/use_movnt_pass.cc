// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: 
//     Guowei Yang <471184555@qq.com>
//     Dun Liang <randonlang@gmail.com>. 
// All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <sstream>
#include "var.h"
#include "opt/pass_manager.h"
#include "opt/pass/use_movnt_pass.h"

namespace jittor {

void UseMovntPass::run() {
    // TODO: need to test this pass
    if (!op->get_loop_option("use_movnt"))
        return;

    for (auto& c : ir->children) {
        if (c->type != "loop") continue;
        c->push_front("//@begin replace \"vmova(.*,.*\\(.*\\))\" \"vmovnt\\g<1>\"", &c->children, true);
        c->push_back("//@end", &c->children, true);
    }
}

} // jittor