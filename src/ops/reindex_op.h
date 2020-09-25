// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "op.h"


namespace jittor {

struct ReindexOp : Op {
    Var* x, * y;
    NanoVector shape;
    vector<string> indexes;
    vector<string> overflow_conditions;
    float64 overflow_value;
    vector<Var*> extras;
    /** 
    Reindex Operator is a one-to-many map operator.
    It performs equivalent Python-pseudo implementation below::

        # input is x, output is y
        n = len(shape)-1
        m = len(x.shape)-1
        k = len(overflow_conditions)-1
        y = np.zeros(shape, x.dtype)
        for i0 in range(shape[0]): # 1-st loop
            for i1 in range(shape[1]): # 2-nd loop
                ...... # many loops
                for in in range(shape[n]) # n+1 -th loop
                    if is_overflow(i0,i1,...,in):
                        y[i0,i1,...,in] = overflow_value
                    else:
                        # indexes[i] is a c++ style integer expression consisting of i0,i1,...,in
                        y[i0,i1,...,in] = x[indexes[0],indexes[1],...,indexes[m]]

        # is_overflow is defined as following
        def is_overflow(i0,i1,...,in):
            return (
                indexes[0] < 0 || indexes[0] >= x.shape[0] ||
                indexes[1] < 0 || indexes[1] >= x.shape[1] ||
                ......
                indexes[m] < 0 || indexes[m] >= x.shape[m] ||

                # overflow_conditions[i] is a c++ style boolean expression consisting of i0,i1,...,in
                overflow_conditions[0] ||
                overflow_conditions[1] ||
                ......
                overflow_conditions[k]
            )
    ----------------
    * [in] x:	A input jittor Var
	
    * [in] shape:	the output shape, a integer array
	
    * [in] indexes:	array of c++ style integer expression, its length should be the same with the number of dimension of x, some buildin variables it can use are::
        
             XDIM, xshape0, ..., xshapen, xstride0, ..., xstriden
             YDIM, yshape0, ..., yshapem, ystride0, ..., ystridem
             i0, i1, ..., in
             @e0(...), @e1(...) for extras input index
             e0p, e1p , ... for extras input pointer
			 
    * [in] overflow_value:	overflow value
	
    * [in] overflow_conditions:	array of c++ style boolean expression, it length can be vary. the buildin variables it can use are the same with indexes
		
    * [in] extras: extra var used for index
	
    ----------------
    Example
    Convolution implemented by reindex operation::

        def conv(x, w):
            N,H,W,C = x.shape
            Kh, Kw, _C, Kc = w.shape
            assert C==_C
            xx = x.reindex([N,H-Kh+1,W-Kw+1,Kh,Kw,C,Kc], [
                'i0', # Nid
                'i1+i3', # Hid+Khid
                'i2+i4', # Wid+KWid
                'i5', # Cid
            ])
            ww = w.broadcast_var(xx)
            yy = xx*ww
            y = yy.sum([3,4,5]) # Kh, Kw, C
            return y, yy
     */
    ReindexOp(Var* x, NanoVector shape, vector<string>&& indexes, float64 overflow_value=0, vector<string>&& overflow_conditions={}, vector<Var*>&& extras={});
    /** Alias x.reindex([i,j,k]) -> 
        x.reindex(i.shape, ['@e0(...)','@e1(...)','@e2(...)',], extras=[i,j,k])
     */
    // @pybind(reindex,reindex_var)
    ReindexOp(Var* x, vector<Var*>&& indexes, float64 overflow_value=0, vector<string>&& overflow_conditions={});

    
    const char* name() const override { return "reindex"; }
    VarPtr grad(Var* out, Var* dout, Var* v, int v_index) override;
    void infer_shape() override;
    VarPtr duplicate() override;
    DECLARE_jit_run;
};

} // jittor
