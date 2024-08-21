// ***************************************************************
// Copyright (c) 2019 
//     Guoye Yang <498731903@qq.com>
//     Dun Liang <randonlang@gmail.com>. 
// All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "op.h"

namespace jittor {

struct CuttTestOp : Op {
    Var* output;
    string cmd;

    CuttTestOp(string cmd);
    
    const char* name() const override { return "cutt_test"; }
    DECLARE_jit_run;
};

} // jittor