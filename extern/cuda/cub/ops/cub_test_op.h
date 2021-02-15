// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "op.h"

namespace jittor {

struct CubTestOp : Op {
    Var* output;
    string cmd;

    CubTestOp(string cmd);
    
    const char* name() const override { return "cub_test"; }
    DECLARE_jit_run;
};

} // jittor