// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: 
//     Guoye Yang <498731903@qq.com>
//     Dun Liang <randonlang@gmail.com>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "op.h"


namespace jittor {

struct CubCumsumOp : Op {
    Var* x, * y;
    bool reverse;

    CubCumsumOp(Var* x, bool reverse=false);
    VarPtr grad(Var* out, Var* dout, Var* v, int v_index) override;
    
    void infer_shape() override;    
    const char* name() const override { return "cub_cumsum"; }
    DECLARE_jit_run;
};

} // jittor