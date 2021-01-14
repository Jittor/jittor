// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <sstream>
#include "var.h"
#include "opt/pass_manager.h"
#include "opt/pass/remove_loop_pass.h"

namespace jittor {

void RemoveLoopPass::run() {
    int loop_id=0;
    for (size_t i=0; i<ir->children.size(); i++) {
        auto& c = ir->children[i];
        if (c->type == "loop") {
            auto choice = op->get_loop_option("remove"+S(loop_id));
            if (choice) {
                c->erase();
                i--;
            }
            loop_id++;
        }
    }
}

} // jittor