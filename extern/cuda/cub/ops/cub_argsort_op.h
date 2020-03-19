// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: 
//     Guoye Yang <498731903@qq.com>
//     Dun Liang <randonlang@gmail.com>. 
// All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "op.h"


namespace jittor {

struct CubArgsortOp : Op {
    Var* x, * indexes, * offsets, * y, * y_key;
    bool descending;
    // @attrs(multiple_outputs)
    CubArgsortOp(Var* x, Var* indexes, Var* offsets, bool descending=false, NanoString dtype=ns_int32);
    VarPtr grad(Var* out, Var* dout, Var* v, int v_index) override;
    void infer_shape() override;
    
    const char* name() const override { return "cub_argsort"; }
    DECLARE_jit_run;
};

} // jittor