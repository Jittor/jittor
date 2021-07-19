// ***************************************************************
// Copyright (c) 2021 Jittor.  All Rights Reserved.
// Maintainers: Dun Liang <randonlang@gmail.com>.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "op.h"
#include "var_slices.h"

namespace jittor {

struct SetitemOp : Op {
    static constexpr int _data_inplaced = NodeFlags::_has_vary_input + 1;
    VarSlices vs;
    // map i to related var slice
    NanoVector i_to_vs;
    // map i to related o
    NanoVector i_to_o;
    NanoVector o_shape;
    int first_oid_of_var, var_dim;
    int bmask;
    NanoString op;

    SetitemOp(Var* x, VarSlices&& slices, Var* y, NanoString op=ns_void);
    
    const char* name() const override { return "setitem"; }
    VarPtr grad(Var* out, Var* dout, Var* v, int v_index) override;
    void infer_shape() override;
    void compile_optimize(string& src) override;
    void graph_optimize() override;
    DECLARE_jit_run;
};

} // jittor
