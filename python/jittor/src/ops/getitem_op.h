// ***************************************************************
// Copyright (c) 2021 Jittor.  All Rights Reserved.
// Maintainers: Dun Liang <randonlang@gmail.com>.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "op.h"
#include "var_slices.h"
#include "misc/stack_vector.h"

namespace jittor {

struct GetitemOp : Op {
    VarSlices vs;
    // map i to related var slice
    NanoVector i_to_vs;
    // map i to related o
    NanoVector i_to_o;
    NanoVector o_shape;
    int first_oid_of_var, var_dim;

    GetitemOp(Var* x, VarSlices&& slices);
    
    const char* name() const override { return "getitem"; }
    VarPtr grad(Var* out, Var* dout, Var* v, int v_index) override;
    void infer_shape() override;
    void compile_optimize(string& src) override;
    void graph_optimize() override;
    DECLARE_jit_run;

    void infer_slices(
        StackVector<>& __restrict__ i_to_vs, 
        StackVector<>& __restrict__ i_to_o,
        StackVector<>& __restrict__ out_shape
    );
    void _compile_optimize(string& src);
};

void cuda_loop_schedule(NanoVector o_shape, int* masks, int* tdims);

} // jittor
