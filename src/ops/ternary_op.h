// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "core/op.h"

namespace jittor {

struct TernaryOp : Op {
    static constexpr bool accepts_storage_strides = true;
    static constexpr bool accepts_cpu_scalar_operands = true;
    Var* cond, * x, * y, * z;
    TernaryOp(Var* cond, Var* x, Var* y);
    
    const char* name() const override { return "ternary"; }
    VarPtr grad(Var* out, Var* dout, Var* v, int v_index) override;
    void infer_shape() override;
    DECLARE_jit_run;
};

} // jittor
