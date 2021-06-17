// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: 
//     Guowei Yang <471184555@qq.com>
//     Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "opt/pass/pass.h"
#include "opt/expr.h"

namespace jittor {

struct MLUPass : Pass {
    MLUPass() : Pass("mark_raw") {};
    void add_memcpy(KernelIR* loop_father, KernelIR* loop, vector<string> vars, vector<string> types, vector<int> is_input, string new_id, vector<string> &nram_vars, vector<string> &nram_types);
    
    int getConvertType(string a, string b);
    int bang_dfs(unique_ptr<KernelIR>& func, string dst, unique_ptr<expr::Expr>& rval, vector<string>& define_vars, vector<string> &bang_code, string new_range);
    int check_int();
    void convert_to_bang(unique_ptr<KernelIR>& func, KernelIR* loop, vector<string> vars, string new_id, string new_range);
    void run() override;
};

} // jittor
