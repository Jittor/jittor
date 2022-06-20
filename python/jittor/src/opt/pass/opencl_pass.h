// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: 
//     Guowei Yang <471184555@qq.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "opt/pass/pass.h"
#include "opt/expr.h"

namespace jittor {

struct OpenclPass : Pass {
    OpenclPass() : Pass("mark_raw") {};

    void add_parentheses(string& str);
    void solve_kernel(KernelIR* c);
    void run() override;
};

} // jittor
