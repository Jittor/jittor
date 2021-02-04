// ***************************************************************
// Copyright (c) 2021 
//     Dun Liang <randonlang@gmail.com>. 
// All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "op.h"

namespace jittor {

struct MpiTestOp : Op {
    Var* output;
    string cmd;

    MpiTestOp(string cmd);
    
    const char* name() const override { return "mpi_test"; }
    DECLARE_jit_run;
};

} // jittor