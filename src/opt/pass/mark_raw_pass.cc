// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <sstream>
#include "var.h"
#include "opt/pass_manager.h"
#include "opt/pass/mark_raw_pass.h"

namespace jittor {

void MarkRawPass::run() {
    vector<string> raws = {"relay_groups"};
    for (auto& c : ir->children) {
        string* check = nullptr;
        bool found = false;
        if (c->type == "define") {
            check = &c->get_attr("rvalue");
        } else if (c->has_attr("code"))
            check = &c->get_attr("code");
        if (check) {
            for (auto& s : raws)
                if (check->find(s) != string::npos) {
                    found = true;
                    break;
                }
            if (found) {
                c->attrs["raw"] = "1";
                if (c->type=="define")
                    raws.push_back(c->get_attr("lvalue"));
            }
        }
    }
}

} // jittor