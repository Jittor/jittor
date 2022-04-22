// ***************************************************************
// Copyright (c) 2022 Jittor. All Rights Reserved. 
// Maintainers: 
//     Guoye Yang <498731903@qq.com>. 
//     Dun Liang <randonlang@gmail.com>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "op.h"

namespace jittor {

//TODO: support FFT2D only now.
struct CufftFftOp : Op {
    bool inverse;
    Var* x, * y;
    NanoString type;
    CufftFftOp(Var* x, bool inverse=false);
    VarPtr grad(Var* out, Var* dout, Var* v, int v_index) override;
    
    const char* name() const override { return "cufft_fft"; }
    DECLARE_jit_run;
};

} // jittor