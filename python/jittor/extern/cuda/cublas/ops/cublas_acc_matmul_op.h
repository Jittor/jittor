// ***************************************************************
// Copyright (c) 2022 Jittor. All Rights Reserved. 
// Maintainers: 
//     Guoye Yang <498731903@qq.com>
//     Dun Liang <randonlang@gmail.com>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "op.h"

namespace jittor {

struct CublasAccMatmulOp : Op {
    Var* a, * b, * c;
    bool trans_a, trans_b;
    int stride_a, stride_b;
    int offset_a, offset_b;
    CublasAccMatmulOp(Var* a, Var* b, bool trans_a, bool trans_b, int stride_a=-1, int stride_b=-1, int offset_a=0, int offset_b=0);
    
    const char* name() const override { return "cublas_acc_matmul"; }
    void infer_shape() override;
    DECLARE_jit_run;
};

} // jittor