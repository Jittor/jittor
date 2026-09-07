// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "op.h"

namespace jittor {

struct ReinterpretViewOp : Op {
    Var* x, * y;
    NanoVector shape;
    NanoString dtype;

    /**
    Returns a tensor that shares the same storage as input but reinterprets its
    dtype and shape. The total byte size must stay unchanged.
     */
    ReinterpretViewOp(Var* x, NanoVector shape, NanoString dtype);

    const char* name() const override { return "reinterpret_view"; }
    VarPtr grad(Var* out, Var* dout, Var* v, int v_index) override;
    void infer_shape() override;
};

} // jittor
