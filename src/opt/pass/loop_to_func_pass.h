// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "opt/pass/pass.h"

namespace jittor {

struct LoopToFuncPass : Pass {
    LoopToFuncPass() : Pass("loop_to_func") {
        reads = {kir::vectorized, kir::unrolled, kir::resplited, kir::lvalue, kir::raw, kir::code, kir::rvalue, kir::dtype};
        writes = {kir::loop_func, kir::dtype, kir::lvalue, kir::code};
    };
    void run() override;
};

} // jittor
