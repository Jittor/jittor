// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// Maintainers: Dun Liang <randonlang@gmail.com>.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "opt/pass/pass.h"

namespace jittor {

struct ReduceAccumulatorPass : Pass {
    ReduceAccumulatorPass() : Pass("reduce_accumulator") {
        reads = {kir::code, kir::dtype, kir::lvalue, kir::rvalue};
        writes = {kir::code};
    };
    void run() override;
};

} // jittor
