// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "codegen/opt/pass/pass.h"

namespace jittor {

struct FusedOp;

// The CUDA block width the generated kernels are launched with, and the value
// ParallelPass writes into their `__launch_bounds__`.  Any pass that reshapes a
// launch has to ask for it here rather than repeat a literal: a block wider than
// the `__launch_bounds__` the kernel was compiled with fails to launch at all.
int cuda_block_width(FusedOp* op);

struct ParallelPass : Pass {
    ParallelPass() : Pass("parallel") {
        reads = {kir::rvalue, kir::code, kir::lvalue, kir::rvalue2, kir::loop_func, kir::loop_id, kir::dtype};
        writes = {kir::rely, kir::dtype, kir::code, kir::rvalue};
    };
    void run() override;
};

} // jittor
