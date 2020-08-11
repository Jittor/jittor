// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
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

    * [in] dtype:   type of return indexes
    
    * [out] out:  return an array of indexes, same length with number of dims of cond 
    
    Example::

        jt.where([[0,0,1],[1,0,0]])
        # return ( [0,2], [1,0] )
     */
    // @attrs(multiple_outputs)
    WhereOp(Var* cond, NanoString dtype=ns_int32);
    void infer_shape() override;
    
    const char* name() const override { return "where"; }
    DECLARE_jit_run;
};

} // jittor