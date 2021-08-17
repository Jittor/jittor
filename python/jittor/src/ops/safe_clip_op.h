// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "op.h"


namespace jittor {

struct SafeClipOp : Op {
    Var* x, * y;
    float64 left, right;
    /** Safe clip value to a range, and keep 
 the gradient pass thought.
 
    * [in] x:   input value
    * [in] left: float64 clip min value.
    * [in] right: float64 clip max value.

     */
    // @pybind(safe_clip)
    SafeClipOp(Var* x, float64 left, float64 right);
    
    const char* name() const override { return "safe_clip"; }
    VarPtr grad(Var* out, Var* dout, Var* v, int v_index) override;
    void infer_shape() override;
    DECLARE_jit_run;
};

} // jittor