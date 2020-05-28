// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "op.h"

namespace jittor {

struct CodeOp : Op {
    vector<Var*> _inputs;
    vector<Var*> _outputs;
    string cpu_src;
    vector<string> cpu_grad_src;
    string cpu_header;
    string cuda_src;
    vector<string> cuda_grad_src;
    string cuda_header;
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
    
        a = jt.random([10])
        b = jt.code(a.shape, "float32", [a],
            cpu_src='''
                for (int i=0; i<in0_shape0; i++)
                    @out(i) = @in0(i)*@in0(i)*2;
            ''',
            cpu_grad_src = ['''
                for (int i=0; i<in0_shape0; i++)
                    @out(i) = @dout(i)*@in0(i)*4;
            '''])

    Example-2::

        a = jt.array([3,2,1])
        b = jt.code(a.shape, a.dtype, [a],
            cpu_header="""
                #include <algorithm>
                @alias(a, in0)
                @alias(b, out)
            """,
            cpu_src="""
                for (int i=0; i<a_shape0; i++)
                    @b(i) = @a(i);
                std::sort(&@b(0), &@b(in0_shape0));
            """
        )
        assert (b.data==[1,2,3]).all()

    Example-3::

        #This example shows how to set multiple outputs in code op.
        a = jt.array([3,2,1])
        b,c = jt.code([(1,), (1,)], [a.dtype, a.dtype], [a],
            cpu_header="""
                #include <iostream>
                using namespace std;
            """,
            cpu_src="""
                @alias(a, in0)
                @alias(b, out0)
                @alias(c, out1)
                @b(0) = @c(0) = @a(0);
                for (int i=0; i<a_shape0; i++) {
                    @b(0) = std::min(@b(0), @a(i));
                    @c(0) = std::max(@c(0), @a(i));
                }
                cout << "min:" << @b(0) << " max:" << @c(0) << endl;
            """
        )
        assert b.data == 1, b
        assert c.data == 3, c

    Example-4::

        #This example shows how to use dynamic shape of jittor variables.
        a = jt.array([5,-4,3,-2,1])
        
        # negtive shape for max size of vary dimension
        b,c = jt.code([(-5,), (-5,)], [a.dtype, a.dtype], [a],
            cpu_src="""
                @alias(a, in0)
                @alias(b, out0)
                @alias(c, out1)
                int num_b=0, num_c=0;
                for (int i=0; i<a_shape0; i++) {
                    if (@a(i)>0)
                        @b(num_b++) = @a(i);
                    else
                        @c(num_c++) = @a(i);
                }
                b->set_shape({num_b});
                c->set_shape({num_c});
            """
        )
        assert (b.data == [5,3,1]).all()
        assert (c.data == [-4,-2]).all()


    CUDA Example-1::

        #This example shows how to use CUDA in code op.
        a = jt.random([100000])
        b = jt.random([100000])
        c = jt.code(a.shape, a.dtype, [a,b],
            cuda_src='''
                __global__ static void kernel1(@ARGS_DEF) {
                    @PRECALC
                    int i = threadIdx.x + blockIdx.x * blockDim.x;
                    int stride = blockDim.x * gridDim.x;
                    for (; i<in0_shape0; i+=stride)
                        @out(i) = @in0(i)*@in1(i);
                }
                kernel1<<<(in0_shape0-1)/1024+1, 1024>>>(@ARGS);
            ''',
            cuda_grad_src = ['''
                __global__ static void kernel2(@ARGS_DEF) {
                    @PRECALC
                    int i = threadIdx.x + blockIdx.x * blockDim.x;
                    int stride = blockDim.x * gridDim.x;
                    for (; i<in0_shape0; i+=stride)
                        @out(i) = @dout(i)*@in1(i);
                }
                kernel2<<<(in0_shape0-1)/1024+1, 1024>>>(@ARGS);
            ''', '''
                __global__ static void kernel3(@ARGS_DEF) {
                    @PRECALC
                    int i = threadIdx.x + blockIdx.x * blockDim.x;
                    int stride = blockDim.x * gridDim.x;
                    for (; i<in0_shape0; i+=stride)
                        @out(i) = @dout(i)*@in0(i);
                }
                kernel3<<<(in0_shape0-1)/1024+1, 1024>>>(@ARGS);
            '''])

    CUDA Example-2::
    
        #This example shows how to use multi dimension data with CUDA.
        a = jt.random((100,100))
        b = jt.random((100,100))
        c = jt.code(a.shape, a.dtype, [a,b],
            cuda_src='''
                __global__ static void kernel1(@ARGS_DEF) {
                    @PRECALC
                    for (int i=blockIdx.x; i<in0_shape0; i+=gridDim.x)
                    for (int j=threadIdx.x; j<in0_shape1; j+=blockDim.x)
                        @out(i,j) = @in0(i,j)*@in1(i,j);
                }
                kernel1<<<32, 32>>>(@ARGS);
            ''',
            cuda_grad_src = ['''
                __global__ static void kernel2(@ARGS_DEF) {
                    @PRECALC
                    for (int i=blockIdx.x; i<in0_shape0; i+=gridDim.x)
                    for (int j=threadIdx.x; j<in0_shape1; j+=blockDim.x)
                        @out(i,j) = @dout(i,j)*@in1(i,j);
                }
                kernel2<<<32, 32>>>(@ARGS);
            ''', '''
                __global__ static void kernel3(@ARGS_DEF) {
                    @PRECALC
                    for (int i=blockIdx.x; i<in0_shape0; i+=gridDim.x)
                    for (int j=threadIdx.x; j<in0_shape1; j+=blockDim.x)
                        @out(i,j) = @dout(i,j)*@in0(i,j);
                }
                kernel3<<<32, 32>>>(@ARGS);
            '''])
     */
    CodeOp(NanoVector shape, NanoString dtype, vector<Var*>&& inputs={}, string&& cpu_src="", vector<string>&& cpu_grad_src={}, string&& cpu_header="", string&& cuda_src="", vector<string>&& cuda_grad_src={}, string&& cuda_header="");

    // @attrs(multiple_outputs)
    CodeOp(vector<NanoVector>&& shapes, vector<NanoString>&& dtypes, vector<Var*>&& inputs={}, string&& cpu_src="", vector<string>&& cpu_grad_src={}, string&& cpu_header="", string&& cuda_src="", vector<string>&& cuda_grad_src={}, string&& cuda_header="");

    const char* name() const override { return "code"; }
    VarPtr grad(Var* out, Var* dout, Var* v, int v_index) override;
    DECLARE_jit_run;
};

} // jittor