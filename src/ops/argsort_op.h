// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: 
//     Guoye Yang <498731903@qq.com>
//     Dun Liang <randonlang@gmail.com>. 
// All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "op.h"


namespace jittor {

struct ArgsortOp : Op {
    Var* x, * y, * y_key;
    string cmp;
    int dim;
    bool descending;
    /** 
    Argsort Operator Perform an indirect sort by given key or compare function.

    x is input, y is output index, satisfy:

        x[y[0]] <= x[y[1]] <= x[y[2]] <= ... <= x[y[n]]

    or

        key(y[0]) <= key(y[1]) <= key(y[2]) <= ... <= key(y[n])

    or

        compare(y[0], y[1]) && compare(y[1], y[2]) && ...

    * [in] x: input var for sort

    * [in] dim: sort alone which dim

    * [in] dtype: type of return indexes

    * [in] key: code for sorted key

    * [in] compare: code for compare

    * [out] index: index have the same size with sorted dim

        
    Example::

            jt.sort([11,13,12])
            # return [0,2,1]
            jt.sort([11,13,12], key='-@x(i)')
            # return [1,2,0]
            jt.sort([11,13,12], key='@x(i)<@x(j)')
            # return [0,2,1]

     */
    // @attrs(multiple_outputs)
    ArgsortOp(Var* x, int dim=-1, bool descending=false, NanoString dtype=ns_int32);
    VarPtr grad(Var* out, Var* dout, Var* v, int v_index) override;
    static VarPtr get_grad(Var* out, Var* dout, Var* v, int v_index, int dim, Var* y);
    void infer_shape() override;
    
    const char* name() const override { return "argsort"; }
    DECLARE_jit_run;
};

} // jittor