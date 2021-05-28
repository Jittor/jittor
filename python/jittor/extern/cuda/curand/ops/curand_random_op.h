// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: 
//     Guoye Yang <498731903@qq.com>. 
//     Dun Liang <randonlang@gmail.com>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "op.h"

namespace jittor {

struct CurandRandomOp : Op {
    Var* output;
    NanoString type;
    CurandRandomOp(NanoVector shape, NanoString dtype=ns_float32, NanoString type=ns_uniform);
    
    const char* name() const override { return "curand_random"; }
    DECLARE_jit_run;
};

} // jittor