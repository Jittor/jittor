// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "opt/pass/pass.h"

namespace jittor {

struct ParallelPass : Pass {
    ParallelPass() : Pass("parallel") {
        reads = {kir::rvalue, kir::code, kir::lvalue, kir::rvalue2, kir::loop_func, kir::loop_id, kir::dtype};
        writes = {kir::rely, kir::dtype, kir::code, kir::rvalue};
    };
    void run() override;
};

} // jittor
