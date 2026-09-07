// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "opt/pass/pass.h"

namespace jittor {

struct WarpReducePass : Pass {
    WarpReducePass() : Pass("warp_reduce") {
        reads = {kir::code};
        writes = {kir::code};
    };
    void run() override;
};

} // jittor
