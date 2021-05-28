// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
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

struct ArgReduceOp : Op {
    Var* x, * y, * y_key;
    NanoString op;
    int dim;
    bool keepdims;

    /**
    Returns the indices of the maximum / minimum of the input across a dimension.

    ----------------

    * [in] x:       the input jt.Var.

    * [in] op:      "max" or "min". 

    * [in] dim:     int. Specifies which dimension to be reduced.

    * [in] keepdim: bool. Whether the output has ``dim`` retained or not.

    ----------------

    Example-1::
        >>> x = jt.randint(0, 10, shape=(2, 3))
        >>> x
        jt.Var([[4 2 5]
         [6 7 1]], dtype=int32)
        >>> jt.arg_reduce(x, 'max', dim=1, keepdims=False)
        [jt.Var([2 1], dtype=int32), jt.Var([5 7], dtype=int32)]
        >>> jt.arg_reduce(x, 'min', dim=1, keepdims=False)
        [jt.Var([1 2], dtype=int32), jt.Var([5 7], dtype=int32)]
     */
    // @attrs(multiple_outputs)
    ArgReduceOp(Var* x, NanoString op, int dim, bool keepdims);
    VarPtr grad(Var* out, Var* dout, Var* v, int v_index) override;
    static VarPtr get_grad(Var* out, Var* dout, Var* v, int v_index, int dim, Var* y);
    void infer_shape() override;
    
    const char* name() const override { return "arg_reduce"; }
    DECLARE_jit_run;
};

} // jittor
