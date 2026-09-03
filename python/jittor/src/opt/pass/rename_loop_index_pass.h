// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "opt/pass/pass.h"

namespace jittor {

struct RenameLoopIndexPass : Pass {
    RenameLoopIndexPass() : Pass("rename_loop_index") {
        reads = {kir::rvalue, kir::lvalue};
        writes = {kir::loop_id, kir::lvalue};
    };
    void run() override;
};

} // jittor
