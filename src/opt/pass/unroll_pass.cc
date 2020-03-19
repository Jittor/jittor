// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <sstream>
#include "var.h"
#include "opt/pass_manager.h"
#include "opt/pass/unroll_pass.h"

namespace jittor {

void UnrollPass::run() {
    auto choice = op->get_loop_option("unroll");
    if (!choice) return;
    vector<KernelIR*> q({ir});
    vector<KernelIR*> loops;
    for (uint i=0; i<q.size(); i++) {
        KernelIR* ir = q[i];
        bool dont_unroll = false;
        // do not unroll if stride != 1
        if (ir->has_attr("rvalue2"))
            dont_unroll = true;
        for (auto& c : ir->children) {
            // non vectorized loop
            if (c->type == "if")
                dont_unroll = true;
            if (c->type == "loop" && !c->has_attr("vectorized") && !c->has_attr("unrolled"))
                dont_unroll = true;
            q.push_back(c.get());
        }
        ASSERT(!(ir->type=="loop" && !dont_unroll && !ir->has_attr("loop_id")));
        if (!dont_unroll && ir->has_attr("loop_id")) {
            loops.push_back(ir);
        }
    }
    for (auto loop : loops) {
        if (loop->has_attr("vectorized") || loop->has_attr("unrolled"))
            continue;
        loop->attrs["unrolled"] = "1";
        if (choice==1)
            loop->push_back("#pragma unroll", &loop->before);
        else {
            int num;
            auto& split_id = loop->get_attr("split_id");
            auto& loop_id = loop->get_attr("loop_id");
            auto& rvalue = loop->get_attr("rvalue");
            if (!loop->get_number(rvalue, num)) {
                if (split_id.size()) {
                    string& si = split_id;
                    ASSERT(loop->get_number("stride"+si, num));
                    if (num>128) {
                        loop->push_back("#pragma unroll", &loop->before);
                        continue;
                    }
                    auto floop = loop->father;
                    while (floop && !floop->check_attr("loop_id", split_id))
                        floop = floop->father;
                    ASSERT(floop) << loop->to_string();
                    floop->resplit();
                    // fully unrolled loops
                    auto loops2 = floop->find_loops(loop_id);
                    ASSERT(loops2.size());
                    for (auto loop2 : loops2) {
                        loop2->before.clear();
                        loop2->push_back("#pragma unroll("+S(num)+")", &loop2->before);
                        loop2->attrs["unrolled"] = "1";
                    }
                    // partial unrolled loops in if
                    ASSERT(floop->after.size() && floop->after[0]->type == "if");
                    auto loops = floop->after[0]->find_loops(loop_id);
                    ASSERT(loops.size());
                    for (auto loop2 : loops) {
                        loop2->before.clear();
                        loop2->push_back("#pragma unroll", &loop2->before);
                        loop2->attrs["unrolled"] = "1";
                    }
                    continue;
                } else {
                    loop->push_back("#pragma unroll", &loop->before);
                    continue;
                }
            }
            loop->push_back("#pragma unroll("+S(num)+")", &loop->before);
        }
    }
}

} // jittor