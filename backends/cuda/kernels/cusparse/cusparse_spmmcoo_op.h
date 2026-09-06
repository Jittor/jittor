// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers:  Shizhan Lu <578752274@qq.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "op.h"
#include "cusparse.h"
namespace jittor {

struct CusparseSpmmcooOp : Op {
    Var* x;
    Var* outputVar;
    Var* row_indices;
    Var* col_indices;
    Var* value;
    Var* output;
    int A_row;
    int A_col;
    bool trans_A;
    bool trans_B;
    CusparseSpmmcooOp(Var* outputVar_, Var* x_, Var* row_indices_,Var* col_indices_,Var* value_,int A_row,int A_col,bool trans_A,bool trans_B);
    const char* name() const override { return "cusparse_spmmcoo"; }
    DECLARE_jit_run;
};

} // jittor