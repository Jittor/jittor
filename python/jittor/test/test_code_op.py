# ***************************************************************
# Copyright (c) 2021 Jittor. All Rights Reserved. 
# Maintainers: Dun Liang <randonlang@gmail.com>. 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
import numpy as np
from jittor import Function

class TestCodeOp(unittest.TestCase):
    def test(self):
        a = jt.random([10])
        b = jt.code(a.shape, a.dtype, [a],
            cpu_src='''
                for (int i=0; i<in0_shape0; i++)
                    @out(i) = @in0(i)*@in0(i)*2;
            ''',
            cpu_grad_src = ['''
                for (int i=0; i<in0_shape0; i++) {
                    @out(i) = @dout(i)*@in0(i)*4;
                }
            '''])
        na, nb = jt.fetch_sync([a,b])
        assert np.allclose(na*na*2, nb)
        
        c = jt.random([10])
        da = jt.grad(c*b, a)
        assert np.allclose(c.data*na*4, da.data), (c.data*na*4, da.data)

    def test_use_func(self):
        class Func(Function):
            def execute(self, x):
                self.save_vars = x
                return jt.code(x.shape, x.dtype, [x],
                    cpu_src='''
                        for (int i=0; i<in0_shape0; i++)
                            @out(i) = @in0(i)*@in0(i)*2;
                    ''')

            def grad(self, grad_x):
                x = self.save_vars
                return jt.code(x.shape, x.dtype, [x, grad_x],
                    cpu_src='''
                        for (int i=0; i<in0_shape0; i++)
                            @out(i) = @in1(i)*@in0(i)*4;
                    ''')

        a = jt.random([10])
        func = Func()
        b = func(a)

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
                for (int i=0; i<in0_shape0; i++)
                    @out(i) = @in0(i)*@in1(i);
            ''',
            cpu_grad_src = ['''
                for (int i=0; i<in0_shape0; i++) {
                    @out(i) = @dout(i)*@in1(i);
                }
            ''', '''
                for (int i=0; i<in0_shape0; i++) {
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

    def test_multi_output(self):
        a = jt.array([3,2,1])
        b,c = jt.code([[2],[4]], ["float32", "float64"], [a],
            cpu_src="""
                @alias(a, in0)
                @alias(b, out0)
                @alias(c, out1)
                for (int i=0; i<a_shape0; i++) {
                    if (i<b_shape0) @b(i) = @a(i);
                    if (i<c_shape0) @c(i) = @a(i);
                }
            """
        )
        assert b.shape == [2]
        assert c.shape == [4]
        assert b.dtype == "float32"
        assert c.dtype == "float64"
        assert (b.data == [3,2]).all()
        assert (c.data[:3] == [3,2,1]).all()

    def test_return_multi_output(self):
        a = jt.array([3,2,1])
        b = jt.array([1,2])
        c = jt.array([3,4,5,6])
        jt.code([a], [b,c],
            cpu_src="""
                @alias(a, in0)
                @alias(b, out0)
                @alias(c, out1)
                for (int i=0; i<a_shape0; i++) {
                    if (i<b_shape0) @b(i) += @a(i);
                    if (i<c_shape0) @c(i) += @a(i);
                }
            """
        )
        assert b.shape == [2]
        assert c.shape == [4]
        assert (b.data == [4,4]).all()
        assert (c.data[:3] == [6,6,6]).all()

    def test_multi_output2(self):
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

    def test_vary_shape(self):
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

    def test_comment(self):
        a = jt.array([3,2,1])
        b = jt.code(a.shape, a.dtype, [a],
            cpu_header='''
            #include <algorithm>
            // asd
            /* asd
            */
            ''',
            cpu_src="""
                // test comment
                /*
                multi line
                */
                @alias(a, in0)
                for (int i=0; i<a_shape0; i++)
                    @out(i) = @a(i);
                std::sort(&@out(0), &@out(a_shape0));
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
                    for (int i=blockIdx.x; i<in0_shape0; i+=gridDim.x)
                    for (int j=threadIdx.x; j<in0_shape1; j+=blockDim.x)
                        @out(i,j) = @in0(i,j)*@in1(i,j);
                }
                kernel1<<<32, 32>>>(@ARGS);
            ''',
            cuda_grad_src = ['''
                __global__ static void kernel(@ARGS_DEF) {
                    @PRECALC
                    for (int i=blockIdx.x; i<in0_shape0; i+=gridDim.x)
                    for (int j=threadIdx.x; j<in0_shape1; j+=blockDim.x)
                        @out(i,j) = @dout(i,j)*@in1(i,j);
                }
                kernel<<<32, 32>>>(@ARGS);
            ''', '''
                __global__ static void kernel(@ARGS_DEF) {
                    @PRECALC
                    @pout(0,0);
                    for (int i=blockIdx.x; i<in0_shape0; i+=gridDim.x)
                    for (int j=threadIdx.x; j<in0_shape1; j+=blockDim.x)
                        @out(i,j) = @dout(i,j)*@in0(i,j);
                }
                kernel<<<32, 32>>>(@ARGS);
            '''])
        da, db = jt.grad(c, [a, b])
        assert np.allclose(c.data, a.data*b.data), (c.data, a.data*b.data)
        assert np.allclose(da.data, b.data)
        assert np.allclose(db.data, a.data)

    @unittest.skipIf(not jt.compiler.has_cuda, "No CUDA found")
    @jt.flag_scope(use_cuda=1)
    def test_cuda2_use_func(self):
        class Func(Function):
            def execute(self, a, b):
                self.save_vars = a, b
                return jt.code(a.shape, a.dtype, [a,b],
                    cuda_src='''
                        __global__ static void kernel1(@ARGS_DEF) {
                            @PRECALC
                            for (int i=blockIdx.x; i<in0_shape0; i+=gridDim.x)
                            for (int j=threadIdx.x; j<in0_shape1; j+=blockDim.x)
                                @out(i,j) = @in0(i,j)*@in1(i,j);
                        }
                        kernel1<<<32, 32>>>(@ARGS);
                    ''')

            def grad(self, grad):
                a, b = self.save_vars
                return jt.code([a.shape, b.shape], [a.dtype, b.dtype], [a, b, grad],
                    cuda_src='''
                        __global__ static void kernel2(@ARGS_DEF) {
                            @PRECALC
                            for (int i=blockIdx.x; i<in0_shape0; i+=gridDim.x)
                            for (int j=threadIdx.x; j<in0_shape1; j+=blockDim.x) {
                                @out0(i,j) = @in2(i,j)*@in1(i,j);
                                @out1(i,j) = @in2(i,j)*@in0(i,j);
                            }
                        }
                        kernel2<<<32, 32>>>(@ARGS);
                    ''')

        a = jt.random((100,100))
        b = jt.random((100,100))
        
        func = Func()
        c = func(a,b)
        da, db = jt.grad(c, [a, b])
        assert np.allclose(c.data, a.data*b.data), (c.data, a.data*b.data)
        assert np.allclose(da.data, b.data)
        assert np.allclose(db.data, a.data)


if __name__ == "__main__":
    unittest.main()