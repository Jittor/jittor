// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "op.h"


namespace jittor {

struct ReindexReduceOp : Op {
    Var* y, * x;
    NanoVector shape;
    vector<string> indexes;
    vector<string> overflow_conditions;
    vector<Var*> extras;
    /**
    Reindex Reduce Operator is a many-to-one map operator.
    It performs equivalent Python-pseudo implementation below::

        # input is y, output is x
        n = len(y.shape)-1
        m = len(shape)-1
        k = len(overflow_conditions)-1
        x = np.zeros(shape, y.dtype)
        x[:] = initial_value(op)
        for i0 in range(y.shape[0]): # 1-st loop
            for i1 in range(y.shape[1]): # 2-nd loop
                ...... # many loops
                for in in range(y.shape[n]) # n+1 -th loop
                    # indexes[i] is a c++ style integer expression consisting of i0,i1,...,in
                    xi0,xi1,...,xim = indexes[0],indexes[1],...,indexes[m]
                    if not is_overflow(xi0,xi1,...,xim):
                        x[xi0,xi1,...,xim] = op(x[xi0,xi1,...,xim], y[i0,i1,...,in])

        # is_overflow is defined as following
        def is_overflow(xi0,xi1,...,xim):
            return (
                xi0 < 0 || xi0 >= shape[0] ||
                xi1 < 0 || xi1 >= shape[1] ||
                ......
                xim < 0 || xim >= shape[m] ||

                # overflow_conditions[i] is a c++ style boolean expression consisting of i0,i1,...,in
                overflow_conditions[0] ||
                overflow_conditions[1] ||
                ......
                overflow_conditions[k]
            )

    * [in] y:   A input jittor Var
    
    * [in] op:  a string represent the reduce operation type
    
    * [in] shape:   the output shape, a integer array
    
    * [in] indexes: array of c++ style integer expression, its length should be the same with length of shape, some buildin variables it can use are::
    
             XDIM, xshape0, ..., xshapem, xstride0, ..., xstridem
             YDIM, yshape0, ..., yshapen, ystride0, ..., ystriden
             i0, i1, ..., in
             @e0(...), @e1(...) for extras input index
             e0p, e1p , ... for extras input pointer
    
    * [in] overflow_conditions: array of c++ style boolean expression, it length can be vary. the buildin variables it can use are the same with indexes.
    
    * [in] extras:  extra var used for index
    
    Example 

    Pooling implemented by reindex operation::

        def pool(x, size, op):
            N,H,W,C = x.shape
            h = (H+size-1)//size
            w = (W+size-1)//size
            return x.reindex_reduce(op, [N,h,w,C], [
                "i0", # Nid
                f"i1/{size}", # Hid
                f"i2/{size}", # Wid
                "i3", # Cid
            ])
     */
    ReindexReduceOp(Var* y, NanoString op, NanoVector shape, vector<string>&& indexes, vector<string>&& overflow_conditions={}, vector<Var*>&& extras={});
    
    const char* name() const override { return "reindex_reduce"; }
    VarPtr grad(Var* out, Var* dout, Var* v, int v_index) override;
    void infer_shape() override;
    DECLARE_jit_run;
};

} // jittor