// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "op.h"

namespace jittor {

struct CublasTestOp : Op {
    Var* output;
    int size_mult;

    CublasTestOp(int size_mult);
    
    const char* name() const override { return "cublas_test"; }
    DECLARE_jit_run;
};

} // jittor