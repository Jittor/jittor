// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: 
//     Guowei Yang <471184555@qq.com>
//     Dun Liang <randonlang@gmail.com>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "op.h"
#include "numpy_func.h"

namespace jittor {

struct NumpyCodeOp : Op {
    vector<Var*> _inputs;
    vector<Var*> _outputs;
    NumpyFunc forward;
    vector<NumpyFunc> backward;
    NumpyResult _results;

    /**
    Numpy Code Operator for easily customized op.

    ----------------

    * [in] shape:    the output shape, a integer array
    
    * [in] dtype:    the output data type
    
    * [in] inputs:   A list of input jittor Vars

    * [in] forward:  function, represents forward python function

    * [in] backward: A list of function, represents gradiant for each input

    ----------------
    
    Example-1::

        def forward_code(np, data):
            a = data["inputs"][0]
            b = data["outputs"][0]
            np.add(a,a,out=b)

        def backward_code(np, data):
            dout = data["dout"]
            out = data["outputs"][0]
            np.copyto(out, dout*2.0)

        a = jt.random((5,1))
        b = jt.numpy_code(
            a.shape,
            a.dtype,
            [a],
            forward_code,
            [backward_code],
        )

    Example-2::
    
        def forward_code(np, data):
            a,b = data["inputs"]
            c,d = data["outputs"]
            np.add(a,b,out=c)
            np.subtract(a,b,out=d)

        def backward_code1(np, data):
            dout = data["dout"]
            out = data["outputs"][0]
            np.copyto(out, dout)

        def backward_code2(np, data):
            dout = data["dout"]
            out_index = data["out_index"]
            out = data["outputs"][0]
            if out_index==0:
                np.copyto(out, dout)
            else:
                np.negative(dout, out)

        a = jt.random((5,1))
        b = jt.random((5,1))
        c, d = jt.numpy_code(
            [a.shape, a.shape],
            [a.dtype, a.dtype],
            [a, b],
            forward_code,
            [backward_code1,backward_code2],
        )

    */
    NumpyCodeOp(NanoVector shape, NanoString dtype, vector<Var*>&& inputs, NumpyFunc&& forward, vector<NumpyFunc>&& backward);

    // @attrs(multiple_outputs)
    NumpyCodeOp(vector<NanoVector>&& shapes, vector<NanoString>&& dtypes, vector<Var*>&& inputs, NumpyFunc&& forward, vector<NumpyFunc>&& backward);

    NumpyCodeOp(NanoVector shape, NanoString dtype, vector<Var*>&& inputs, NumpyFunc&& forward);

    // @attrs(multiple_outputs)
    NumpyCodeOp(vector<NanoVector>&& shapes, vector<NanoString>&& dtypes, vector<Var*>&& inputs, NumpyFunc&& forward);
    
    // @pybind(None)
    NumpyCodeOp(NanoVector shape, NanoString dtype, vector<Var*>&& inputs, NumpyFunc forward, NumpyResult&& results);

    const char* name() const override { return "numpy_code"; }
    VarPtr grad(Var* out, Var* dout, Var* v, int v_index) override;
    
    void run() override;
};

} // jittor