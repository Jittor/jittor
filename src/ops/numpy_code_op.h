// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: 
//     Guowei Yang <471184555@qq.com>
//     Dun Liang <randonlang@gmail.com>. 
// All Rights Reserved.
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
    Code Operator for easily customized op.

    ----------------

    * [in] shape:   the output shape, a integer array
    
    * [in] dtype:   the output data type
    
    * [in] inputs:  A list of input jittor Vars
    
    * [in] cpu_src: cpu source code string, buildin value:

            *   in{x}, in{x}_shape{y}, in{x}_stride{y}, in{x}_type, in{x}_p, @in0(...)
            *   out{x}, out{x}_shape{y}, out{x}_stride{y}, out{x}_type, out{x}_p, @out0(...)
            *   out, out_shape{y}, out_stride{y}, out_type, out_p, @out(...)
    
    * [in] cpu_grad_src:    A list of string, cpu source code string for gradient, represents gradiant for each inputm buildin value, buildin value:

        *   in{x}, in{x}_shape{y}, in{x}_stride{y}, in{x}_type, in{x}_p, @in0(...)
        *   out{x}, out{x}_shape{y}, out{x}_stride{y}, out{x}_type, out{x}_p, @out0(...)
        *   out, out_shape{y}, out_stride{y}, out_type, out_p, @out(...)
        *   pout{x}, pout{x}_shape{y}, pout{x}_stride{y}, pout{x}_type, pout{x}_p, @pout{x}(...)
        *   pout, pout_shape{y}, pout_stride{y}, pout_type, pout_p, @pout(...)
        *   dout, dout_shape{y}, dout_stride{y}, dout_type, dout_p, @dout(...)
    
    * [in] cpu_header: cpu header code string.

    * [in] cuda_src: cuda source code string.

    * [in] cuda_grad_src:   A list of string.

    * [in] cuda_header: cuda header code string.

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
    
    // @pybind(None)
    NumpyCodeOp(NanoVector shape, NanoString dtype, vector<Var*>&& inputs, NumpyFunc forward, NumpyResult&& results);

    const char* name() const override { return "numpy_code"; }
    VarPtr grad(Var* out, Var* dout, Var* v, int v_index) override;
    
    void run() override;
};

} // jittor