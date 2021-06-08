// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: 
//     Guowei Yang <471184555@qq.com>
//     Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <sstream>
#include "var.h"
#include "opt/pass_manager.h"
#include "opt/pass/mlu_split_pass.h"
#include "misc/mlu_flags.h"
#include "opt/pass/loop_var_analyze_pass.h"
#include "opt/expr.h"

namespace jittor {

int MLUSplitPass::check_int(){
    for (auto& c : ir->father->children)
    if (c->type=="macro") {
        string str=c->get_attr("code");
        if (str.find("#define")==-1 || str.find("_T")==-1) continue;
        if (str.find("int")!=-1) return 1;
    }
    return 0;
}

void set_cannot_parallel(KernelIR* loop) {
    for (auto& cc : loop->children)
    if (cc->type=="loop") {
        cc->attrs["cannot_parallel"]="1";
        return;
    }
}

void MLUSplitPass::run() {
    if (!use_mlu) return;
    
    int num = 0;
    int choice = nram_space;
    KernelIR *loop;
    string i, j;
    for (auto& c : ir->before) {
        if (c->get_attr("dtype") == "INLINE_FUNC") {
            KernelIR* sc=&(*c);
            vector<string> ids;
            while (1){
                int bo=0;
                for (auto& cc : sc->children)
                if (cc->type=="loop") {
                    ids.push_back(cc->inner[0]->get_attr("lvalue"));
                    bo=1;
                    sc = &(*cc);
                    loop = sc;
                    break;
                }
                if (!bo) break;
                else num++;
            }
            i = loop->get_attr("loop_id");
            j = i+"0";
            c->split_loop(i, j);
            set_cannot_parallel(loop);
            c->push_back(loop->attrs["dtype"]+" stride"+i+" = "+S(choice)+";");
            c->move_loop_back();
        }
    }
}

} // jittor