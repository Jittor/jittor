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

struct ReshapeOp : Op {
    Var* x, * y;
    NanoVector shape;

    /**
    Returns a tensor with the same data and number of elements as input, but with the specified shape. 

    A single dimension may be -1, in which case it’s inferred from the remaining dimensions and the number of elements in input.

    ----------------

    * [in] x:       the input jt.Var

    * [in] shape:   the output shape, an integer array

    ----------------

    Example-1::
        >>> a = jt.randint(0, 10, shape=(12,))
        >>> a
        jt.Var([4 0 8 4 6 3 1 8 1 1 2 2], dtype=int32)
        >>> jt.reshape(a, (3, 4))
        jt.Var([[4 0 8 4]
         [6 3 1 8]
         [1 1 2 2]], dtype=int32)
        >>> jt.reshape(a, (-1, 6))
        jt.Var([[4 0 8 4 6 3]
         [1 8 1 1 2 2]], dtype=int32)
     */
    ReshapeOp(Var* x, NanoVector shape);
    
    const char* name() const override { return "reshape"; }
    VarPtr grad(Var* out, Var* dout, Var* v, int v_index) override;
    void infer_shape() override;
};
} // jittor