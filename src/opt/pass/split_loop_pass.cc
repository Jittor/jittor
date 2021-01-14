// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <sstream>
#include "var.h"
#include "opt/pass_manager.h"
#include "opt/pass/split_loop_pass.h"
#include "opt/pass/loop_var_analyze_pass.h"

namespace jittor {

void SplitLoopPass::run() {
    auto* lva_pass = pm->get_pass<LoopVarAnalyzePass>("loop_var_analyze");
    ASSERT(lva_pass);
    if (op->flags.get(NodeFlags::_cpu))
        ir->push_back("using namespace std;", &ir->before);
    number_of_ranges_after_split = lva_pass->number_of_ranges;
    for (int i=0; i<number_of_ranges_after_split; i++) {
        auto choice = op->get_loop_option("split"+S(i));
        if (choice > 1) {
            int j = number_of_ranges_after_split++;
            int split_size = std::max(1, choice);
            auto loops = ir->find_loops(S(i));
            ASSERT(loops.size());
            ir->push_back(loops[0]->attrs["dtype"]+" stride"+S(i)+" = "+S(split_size)+";");
            ir->split_loop(i, j);
        }
    }
    ir->move_loop_back();
}

} // jittor