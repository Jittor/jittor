// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "op.h"


namespace jittor {

struct IndexOp : Op {
    unique_ptr<Var*[]> x;
    int64 dim;
    /** 
    Index Operator generate index of shape.
    
    It performs equivalent Python-pseudo implementation below::
    
        n = len(shape)-1
        x = np.zeros(shape, dtype)
        for i0 in range(shape[0]): # 1-st loop
            for i1 in range(shape[1]): # 2-nd loop
                ...... # many loops
                for in in range(shape[n]) # n+1 -th loop
                    x[i0,i1,...,in] = i@dim
    
    * [in] shape:   the output shape, a integer array
    * [in] dim: the dim of the index.
    * [in] dtype:   the data type string, default int32

    Example::

        print(jt.index([2,2], 0)())
        # output: [[0,0],[1,1]]
        print(jt.index([2,2], 1)())
        # output: [[0,1],[0,1]]
     */
    IndexOp(NanoVector shape, int64 dim, NanoString dtype=ns_int32);
    // @attrs(multiple_outputs)
    IndexOp(NanoVector shape, NanoString dtype=ns_int32);
    /** shape dependency version of index op
        jt.index_var(a, 1) similar with jt.index(a.shape, 1)
     */
    // @pybind(index,index_var)
    IndexOp(Var* a, int64 dim, NanoString dtype=ns_int32);
    /** shape dependency version of index op
        jt.index_var(a) similar with jt.index(a.shape)
     */
    // @pybind(index,index_var)
    // @attrs(multiple_outputs)
    IndexOp(Var* a, NanoString dtype=ns_int32);
    
    const char* name() const override { return "index"; }
    void infer_shape() override;
    DECLARE_jit_run;
};

} // jittor