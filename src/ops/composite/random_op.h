// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "core/op.h"

namespace jittor {

struct RandomOp : Op {
    // Accelerators reach this op either through their Random capability op
    // (CUDA-family curand_random) or, when they run `random` themselves,
    // through a native callback their composer installs on this entry.
    static constexpr uint32 backend_mask = OpBackendAny;
    Var* output;
    NanoString type;
    RandomOp(NanoVector shape, NanoString dtype=ns_float32, NanoString type=ns_uniform);
    
    const char* name() const override { return "random"; }
    DECLARE_jit_run;
};

} // jittor
