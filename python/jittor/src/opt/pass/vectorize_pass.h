// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "opt/pass/pass.h"

namespace jittor {

struct VectorizePass : Pass {
    VectorizePass() : Pass("vectorize") {
        reads = {kir::loop_id, kir::rvalue, kir::split_id};
        writes = {kir::vectorized};
    };
    void run() override;
};

} // jittor
