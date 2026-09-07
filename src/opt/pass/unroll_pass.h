// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "opt/pass/pass.h"

namespace jittor {

struct UnrollPass : Pass {
    UnrollPass() : Pass("unroll") {
        reads = {kir::loop_id, kir::rvalue, kir::rvalue2, kir::split_id, kir::vectorized, kir::unrolled};
        writes = {kir::unrolled, kir::resplited};
    };
    void run() override;
};

} // jittor
