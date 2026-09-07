// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "opt/pass/pass.h"

namespace jittor {

struct MergeLoopVarPass : Pass {
    MergeLoopVarPass() : Pass("merge_loop_var") {
        reads = {kir::rvalue, kir::lvalue, kir::loop_id, kir::dtype};
        writes = {kir::loop_id, kir::rvalue, kir::dtype};
    };
    void run() override;
};

} // jittor
