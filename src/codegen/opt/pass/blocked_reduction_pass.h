// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// Maintainers: Dun Liang <randonlang@gmail.com>.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "codegen/opt/pass/pass.h"

namespace jittor {

struct BlockedReductionPass : Pass {
    BlockedReductionPass() : Pass("blocked_reduction") {
        reads = {kir::code, kir::dtype, kir::lvalue, kir::rvalue,
                 kir::rvalue2, kir::reduce_acc, kir::has_bc};
        writes = {kir::code, kir::raw};
    };
    void run() override;
};

} // jittor
