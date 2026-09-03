// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "opt/pass/pass.h"

namespace jittor {

struct SplitLoopPass : Pass {
    int number_of_ranges_after_split;

    SplitLoopPass() : Pass("split_loop"), number_of_ranges_after_split(0)  {
        reads = {kir::dtype, kir::lvalue, kir::rvalue, kir::loop_id};
        writes = {kir::rvalue2, kir::loop_id, kir::split_id, kir::code};
    };
    void run() override;
};

} // jittor
