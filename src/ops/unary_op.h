// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "op.h"


namespace jittor {

struct UnaryOp : Op {
    Var* x, * y;
    // @pybind(unary,cast)
    UnaryOp(Var* x, NanoString op);
    
    const char* name() const override { return "unary"; }
    VarPtr grad(Var* out, Var* dout, Var* v, int v_index) override;
    void infer_shape() override;
    DECLARE_jit_run;
};

} // jittor