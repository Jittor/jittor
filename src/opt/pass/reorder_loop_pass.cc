// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <sstream>
#include "var.h"
#include "opt/pass_manager.h"
#include "opt/pass/reorder_loop_pass.h"
#include "opt/pass/loop_var_analyze_pass.h"
#include "opt/pass/split_loop_pass.h"

namespace jittor {

vector<int> ReorderLoopPass::search_parse_loop_order() {
    vector<int> order;
    auto* sl_pass = pm->get_pass<SplitLoopPass>("split_loop");
    ASSERT(sl_pass);
    auto number_of_ranges_after_split = sl_pass->number_of_ranges_after_split;
    if (!number_of_ranges_after_split) return order;
    for (int i=0; i<number_of_ranges_after_split; i++) {
        auto choice = op->get_loop_option("order"+S(i));
        ASSERT(choice<=i);
        order.insert(order.end()-choice, i);
    }
    ASSERT(order.size() == (uint)number_of_ranges_after_split);
    return order;
}

void ReorderLoopPass::run() {
    vector<int> order = search_parse_loop_order();
    vector<KernelIR*> loops;
    for (uint i=0; i<ir->children.size(); i++) {
        KernelIR* loop = ir->children[i].get();
        if (loop->type != "loop")
            continue;
        loops.clear();
        loops.push_back(loop);
        while (1) {
            loop = loops.back();
            KernelIR* loop2 = nullptr;
            for (auto& c : loop->children) {
                if (c->type != "loop")
                    continue;
                ASSERT(loop2 == nullptr);
                loop2 = c.get();
            }
            if (loop2 == nullptr) break;
            ASSERT(loop->children.size()==1);
            loops.push_back(loop2);
        }
        // sort loop with order
        int count=0;
        for (auto j : order) {
            uint k;
            for (k=count; k<loops.size(); k++)
                if (loops[k]->check_attr("loop_id", S(j)))
                    break;
            if (k<loops.size())
                loops[k]->swap(*loops[count++]);
        }
    }
}

} // jittor