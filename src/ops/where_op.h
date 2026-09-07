// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "op.h"


namespace jittor {

struct WhereOp : Op {
    Var* cond;
    unique_ptr<Var*[]> outs;
    /**
    Where Operator generate index of true condition.

    * [in] cond:    condition for index generation

    * [in] dtype:   type of return indexes; int64 like torch, so an index can
                    still name an element of a tensor with more than 2**31 of
                    them, and so it survives arithmetic (Jittor promotes by
                    byte width, so `index * stride` stays in the index's dtype)
    
    * [out] out:  return an array of indexes, same length with number of dims of cond 
    
    Example::

        jt.where([[0,0,1],[1,0,0]])
        # return [jt.Var([0 1], dtype=int64), jt.Var([2 0], dtype=int64)]
     */
    // @attrs(multiple_outputs)
    WhereOp(Var* cond, NanoString dtype=ns_int64);
    /**
     * Condition operator, perform cond ? x : y
     * */
    WhereOp(Var* cond, Var* x, Var* y);
    void infer_shape() override;
    
    const char* name() const override { return "where"; }
    DECLARE_jit_run;
};

} // jittor