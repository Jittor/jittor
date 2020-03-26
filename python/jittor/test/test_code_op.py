# ***************************************************************
# Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
import numpy as np

class TestCodeOp(unittest.TestCase):
    def test(self):
        a = jt.random([10])
        b = jt.code(a.shape, a.dtype, [a],
            cpu_src='''
                for (int i=0; i<in0shape0; i++)
                    @out(i) = @in0(i)*@in0(i)*2;
            ''',
            cpu_grad_src = ['''
                for (int i=0; i<in0shape0; i++) {
                    @out(i) = @dout(i)*@in0(i)*4;
                }
            '''])
        na, nb = jt.fetch_sync([a,b])
        assert np.allclose(na*na*2, nb)
        
        c = jt.random([10])
        da = jt.grad(c*b, a)
        assert np.allclose(c.data*na*4, da.data), (c.data*na*4, da.data)

    def test_multi_input(self):
        a = jt.random([10])
        b = jt.random([10])
        c = jt.code(a.shape, a.dtype, [a,b],
            cpu_src='''
                for (int i=0; i<in0shape0; i++)
                    @out(i) = @in0(i)*@in1(i);
            ''',
            cpu_grad_src = ['''
                for (int i=0; i<in0shape0; i++) {
                    @out(i) = @dout(i)*@in1(i);
                }
            ''', '''
                for (int i=0; i<in0shape0; i++) {
                    @out(i) = @dout(i)*@in0(i);
                }
            '''])
        da, db = jt.grad(c, [a, b])
        assert np.allclose(c.data, a.data*b.data), (c.data, a.data*b.data)
        assert np.allclose(da.data, b.data)
        assert np.allclose(db.data, a.data)

    def test_header(self):
        a = jt.array([3,2,1])
        b = jt.code(a.shape, a.dtype, [a],
            cpu_header='#include <algorithm>',
            cpu_src="""
                for (int i=0; i<in0shape0; i++)
                    @out(i) = @in0(i);
                std::sort(&@out(0), &@out(in0shape0));
            """
        )
        assert (b.data==[1,2,3]).all()

    @unittest.skipIf(not jt.compiler.has_cuda, "No CUDA found")
    @jt.flag_scope(use_cuda=1)
    def test_cuda(self):
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
        da, db = jt.grad(c, [a, b])
        assert np.allclose(c.data, a.data*b.data), (c.data, a.data*b.data)
        assert np.allclose(da.data, b.data)
        assert np.allclose(db.data, a.data)

    @unittest.skipIf(not jt.compiler.has_cuda, "No CUDA found")
    @jt.flag_scope(use_cuda=1)
    def test_cuda2(self):
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
                __global__ static void kernel(@ARGS_DEF) {
                    @PRECALC
                    for (int i=blockIdx.x; i<in0shape0; i+=gridDim.x)
                    for (int j=threadIdx.x; j<in0shape1; j+=blockDim.x)
                        @out(i,j) = @dout(i,j)*@in1(i,j);
                }
                kernel<<<32, 32>>>(@ARGS);
            ''', '''
                __global__ static void kernel(@ARGS_DEF) {
                    @PRECALC
                    @pout(0,0);
                    for (int i=blockIdx.x; i<in0shape0; i+=gridDim.x)
                    for (int j=threadIdx.x; j<in0shape1; j+=blockDim.x)
                        @out(i,j) = @dout(i,j)*@in0(i,j);
                }
                kernel<<<32, 32>>>(@ARGS);
            '''])
        da, db = jt.grad(c, [a, b])
        assert np.allclose(c.data, a.data*b.data), (c.data, a.data*b.data)
        assert np.allclose(da.data, b.data)
        assert np.allclose(db.data, a.data)


if __name__ == "__main__":
    unittest.main()