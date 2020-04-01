// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "op.h"

namespace jittor {

struct CodeOp : Op {
    vector<Var*> in;
    Var* out;
    string cpu_src;
    vector<string> cpu_grad_src;
    string cpu_header;
    string cuda_src;
    vector<string> cuda_grad_src;
    string cuda_header;
    /**
    Code Operator for easily customized op.

    ----------------

    @param[in]	shape   the output shape, a integer array
    
    @param[in]	dtype   the output data type
    
    @param[in]	inputs  A list of input jittor Vars
    
    @param[in]	cpu_src cpu source code string, buildin value:
        *   in{x}, in{x}shape{y}, in{x}stride{y}, Tin{x}, in{x}p, @in0(...)
        *   out, outshape{y}, outstride{y}, Tout, outp, @out(...)
    
    @param[in]	cpu_grad_src    A list of string, 
        cpu source code string for gradient, represents gradiant
        for each inputm buildin value, buildin value:
        *   in{x}, in{x}shape{y}, in{x}stride{y}, Tin{x}, in{x}p, @in0(...)
        *   out, outshape{y}, outstride{y}, Tout, outp, @out(...)
        *   pout, poutshape{y}, poutstride{y}, Tpout, poutp, @pout(...)
        *   dout, doutshape{y}, doutstride{y}, Tdout, doutp, @dout(...)
    
    @param[in]	cpu_header cpu header code string.

    @param[in]	cuda_src cuda source code string.

    @param[in]	cuda_grad_src   A list of string.

    @param[in]	cuda_header cuda header code string.
    
    ----------------
    
    Example
    
    ```
    a = jt.random([10])
    b = jt.code(a.shape, a.dtype, [a],
        cpu_src='''
            for (int i=0; i<in0shape0; i++)
                @out(i) = @in0(i)*@in0(i)*2;
        ''',
        cpu_grad_src = ['''
            for (int i=0; i<in0shape0; i++)
                @out(i) = @dout(i)*@in0(i)*4;
        '''])
    ```

    Example2(CUDA):
    ```
    a = jt.random([100000])
    b = jt.random([100000])
    c = jt.code(a.shape, a.dtype, [a,b],
        cuda_src='''
            __global__ static void kernel1(@ARGS_DEF) {
                @PRECALC
                int i = threadIdx.x + blockIdx.x * blockDim.x;
                int stride = blockDim.x * gridDim.x;
                for (; i<in0shape0; i+=stride)
                    @out(i) = @in0(i)*@in1(i);
            }
            kernel1<<<(in0shape0-1)/1024+1, 1024>>>(@ARGS);
        ''',
        cuda_grad_src = ['''
            __global__ static void kernel2(@ARGS_DEF) {
                @PRECALC
                int i = threadIdx.x + blockIdx.x * blockDim.x;
                int stride = blockDim.x * gridDim.x;
                for (; i<in0shape0; i+=stride)
                    @out(i) = @dout(i)*@in1(i);
            }
            kernel2<<<(in0shape0-1)/1024+1, 1024>>>(@ARGS);
        ''', '''
            __global__ static void kernel3(@ARGS_DEF) {
                @PRECALC
                int i = threadIdx.x + blockIdx.x * blockDim.x;
                int stride = blockDim.x * gridDim.x;
                for (; i<in0shape0; i+=stride)
                    @out(i) = @dout(i)*@in0(i);
            }
            kernel3<<<(in0shape0-1)/1024+1, 1024>>>(@ARGS);
        '''])
    ```

    Example3(CUDA):
    ```
    a = jt.random((100,100))
    b = jt.random((100,100))
    c = jt.code(a.shape, a.dtype, [a,b],
        cuda_src='''
            __global__ static void kernel1(@ARGS_DEF) {
                @PRECALC
                for (int i=blockIdx.x; i<in0shape0; i+=gridDim.x)
                for (int j=threadIdx.x; j<in0shape1; j+=blockDim.x)
                    @out(i,j) = @in0(i,j)*@in1(i,j);
            }
            kernel1<<<32, 32>>>(@ARGS);
        ''',
        cuda_grad_src = ['''
            __global__ static void kernel2(@ARGS_DEF) {
                @PRECALC
                for (int i=blockIdx.x; i<in0shape0; i+=gridDim.x)
                for (int j=threadIdx.x; j<in0shape1; j+=blockDim.x)
                    @out(i,j) = @dout(i,j)*@in1(i,j);
            }
            kernel2<<<32, 32>>>(@ARGS);
        ''', '''
            __global__ static void kernel3(@ARGS_DEF) {
                @PRECALC
                for (int i=blockIdx.x; i<in0shape0; i+=gridDim.x)
                for (int j=threadIdx.x; j<in0shape1; j+=blockDim.x)
                    @out(i,j) = @dout(i,j)*@in0(i,j);
            }
            kernel3<<<32, 32>>>(@ARGS);
        '''])
    ```
     */
    CodeOp(NanoVector shape, NanoString dtype, vector<Var*>&& inputs={}, string&& cpu_src="", vector<string>&& cpu_grad_src={}, string&& cpu_header="", string&& cuda_src="", vector<string>&& cuda_grad_src={}, string&& cuda_header="");

    const char* name() const override { return "code"; }
    VarPtr grad(Var* out, Var* dout, Var* v, int v_index) override;
    DECLARE_jit_run;
};

} // jittor