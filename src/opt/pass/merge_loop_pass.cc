// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <sstream>
#include "var.h"
#include "opt/pass_manager.h"
#include "opt/pass/merge_loop_pass.h"

namespace jittor {

void MergeLoopPass::run() {
    auto choice = op->get_loop_option("merge", 1);
    if (!choice) return;
    bool is_cuda = op->flags.get(NodeFlags::_cuda);
    if (is_cuda) {
        vector<KernelIR*> loops;
        vector<string> loop_keys;
        for (auto& c : ir->children) {
            if (c->type != "loop")
                continue;
            if (!c->has_attr("loop_id"))
                continue;
            if (c->has_attr("raw"))
                continue;
            auto* cc = c.get();
            string key = cc->get_attr("loop_id");
            while (cc->children.size()==1 && cc->children[0]->has_attr("loop_id")) {
                cc = cc->children[0].get();
                key += cc->get_attr("loop_id");
            }
            loops.push_back(c.get());
            loop_keys.push_back(key);
        }
        LOGvvvv << "loop keys" << loop_keys;
        for (int i=(int)loops.size()-1; i>=0; i--) {
            if (!loops[i]) continue;
            for (int j=i-1; j>=0; j--) {
                if (!loops[j]) continue;
                int cpx=0; // commen prefix
                auto& ki = loop_keys[i];
                auto& kj = loop_keys[j];
                while (cpx < ki.size() && cpx<kj.size() && ki[cpx] == kj[cpx]) cpx++;
                int mismatch = std::max(ki.size(), kj.size()) - cpx;
                LOGvvvv << "loop key " << ki << kj << "mismatch" << mismatch;
                if (mismatch>=2 || cpx==0)
                    continue;
                loops[i]->insert(0, loops[j]->children);
                loops[i]->merge_loop();
                loops[j]->erase();
                loops[j] = nullptr;
            }
        }
    } else {
        ir->merge_loop();
    }
}

} // jittor