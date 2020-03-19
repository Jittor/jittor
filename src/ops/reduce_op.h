// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "op.h"

namespace jittor {

struct ReduceOp : Op {
    Var* x, * y;
    NanoVector yshape; // keepdim shape
    bool keepdims;
    uint16 reduce_mask; // i-th bit is 1 of dim-i is reduced
    ReduceOp(Var* x, NanoString op, int dim, bool keepdims=false);
    ReduceOp(Var* x, NanoString op, NanoVector dims=NanoVector(), bool keepdims=false);
    // @pybind(None)
    ReduceOp(Var* x, NanoString op, uint dims_mask, bool keepdims);
    
    const char* name() const override { return "reduce"; }
    VarPtr grad(Var* out, Var* dout, Var* v, int v_index) override;
    void infer_shape() override;
    DECLARE_jit_run;
};

} // jittor