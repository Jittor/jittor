// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: 
//     Guoye Yang <498731903@qq.com>. 
//     Dun Liang <randonlang@gmail.com>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "core/op.h"
#include "runtime/random_generator.h"

namespace jittor {

struct CurandRandomOp : Op {
    Var* output;
    NanoString type;
    // Set when a jt.Generator was passed: this draw's key and first position
    // in that generator's counter-based stream, reserved at construction
    // (RandomGeneratorState::reserve_cuda). Without one the op draws from the
    // global curand host generator.
    bool from_generator = false;
    uint64 philox_seed = 0;
    uint64 philox_offset = 0;
    CurandRandomOp(NanoVector shape, NanoString dtype=ns_float32, NanoString type=ns_uniform, RandomGenerator* generator=nullptr);

    const char* name() const override { return "curand_random"; }
    DECLARE_jit_run;
};

} // jittor