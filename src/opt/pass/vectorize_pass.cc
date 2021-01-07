// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <sstream>
#include "var.h"
#include "opt/pass_manager.h"
#include "opt/pass/vectorize_pass.h"

namespace jittor {

void VectorizePass::run() {
    auto choice = op->get_loop_option("vectorize");
    if (!choice) return;
    vector<KernelIR*> q({ir});
    vector<KernelIR*> inner_loops;
    for (uint i=0; i<q.size(); i++) {
        KernelIR* ir = q[i];
        bool has_loop = false;
        for (auto& c : ir->children) {
            if (c->type == "loop")
                has_loop = true;
            q.push_back(c.get());
        }
        if (!has_loop && ir->has_attr("loop_id"))
            inner_loops.push_back(ir);
    }
    LOGvvvv << "Find" << inner_loops.size() << "inner loops";
    for (auto loop : inner_loops) {
        if (choice == 1) {
            loop->push_back("#pragma vector", &loop->before);
        } else if (choice > 1) {
            int num=0;
            if (!loop->get_number(loop->get_attr("rvalue"), num)) {
                if (loop->has_attr("split_id")) {
                    string si = loop->attrs["split_id"];
                    string loop_id = loop->attrs["loop_id"];
                    ASSERT(loop->get_number("stride"+si, num));
                    int vectorlength = 64;
                    while (vectorlength && vectorlength/2 >= num)
                        vectorlength /= 2;
                    auto floop = loop->father;
                    while (floop && !floop->check_attr("loop_id", si))
                        floop = floop->father;
                    ASSERT(floop);
                    auto loops = floop->find_loops(loop_id);
                    ASSERT(loops.size());
                    for (auto loop2 : loops) {
                        loop2->before.clear();
                        loop2->push_back("#pragma vector", &loop2->before);
                        loop2->attrs["vectorized"] = "1";
                    }
                    floop->resplit();
                    auto loops2 = floop->find_loops(loop_id);
                    ASSERT(loops2.size());
                    for (auto loop2 : loops2) {
                        loop2->before.clear();
                        loop2->push_back("#pragma vector vectorlength("+S(vectorlength)+")", &loop2->before);
                        loop2->attrs["vectorized"] = "1";
                    }
                    continue;
                }
                loop->push_back("#pragma vector", &loop->before);
            } else {
                int vectorlength = 64;
                while (vectorlength && vectorlength/2 >= num)
                    vectorlength /= 2;
                if (vectorlength > 1)
                    loop->push_back("#pragma vector vectorlength("+S(vectorlength)+")", &loop->before);
                else
                    loop->push_back("#pragma vector", &loop->before);
            }
        }
        loop->push_back("#pragma ivdep", &loop->before);
        loop->attrs["vectorized"] = "1";
    }
}

} // jittor