// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "core/op.h"

namespace jittor {

struct FuseTransposeOp : Op {
    Var* x, * y;
    NanoVector axes;
    FuseTransposeOp(Var* x, NanoVector axes=NanoVector());
    // For the kernel, which takes the permutation at run time and reaches the
    // op as `y->input()` (a fused kernel binds only an op's Var members):
    // the stride in x of the axis that output axis i reads,
    static int64 input_stride_of_output_axis(Op* op, int i);
    // and the stride in y of the axis that input axis i moves to.
    static int64 output_stride_of_input_axis(Op* op, int i);
    
    const char* name() const override { return "fuse_transpose"; }
    VarPtr grad(Var* out, Var* dout, Var* v, int v_index) override;
    void infer_shape() override;
    DECLARE_jit_run;
};

} // jittor