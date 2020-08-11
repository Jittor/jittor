# ***************************************************************
# Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
from .test_core import expect_error

def test_cuda(use_cuda=1):
    @unittest.skipIf(not jt.compiler.has_cuda, "No CUDA found")
    class TestCudaBase(unittest.TestCase):
        def setUp(self):
            jt.flags.use_cuda = use_cuda
        def tearDown(self):
            jt.flags.use_cuda = 0
    return TestCudaBase

@unittest.skipIf(not jt.compiler.has_cuda, "No CUDA found")
class TestCuda(unittest.TestCase):
    @jt.flag_scope(use_cuda=1)
    def test_cuda_flags(self):
        a = jt.random((10, 10))
        a.sync()

    @jt.flag_scope(use_cuda=2)
    def test_no_cuda_op(self):
        no_cuda_op = jt.compile_custom_op("""
        struct NoCudaOp : Op {
            Var* output;
            NoCudaOp(NanoVector shape, string dtype="float");
            
            const char* name() const override { return "my_cuda"; }
            DECLARE_jit_run;
        };
        """, """
        #ifndef JIT
        NoCudaOp::NoCudaOp(NanoVector shape, string dtype) {
            flags.set(NodeFlags::_cpu);
            output = create_output(shape, dtype);
        }

        void NoCudaOp::jit_prepare() {
            add_jit_define("T", output->dtype());
        }

        #else // JIT
        void NoCudaOp::jit_run() {}
        #endif // JIT
        """,
        "no_cuda")
        # force use cuda
        a = no_cuda_op([3,4,5], 'float')
        expect_error(lambda: a())

    @jt.flag_scope(use_cuda=1)
    def test_cuda_custom_op(self):
        my_op = jt.compile_custom_op("""
        struct MyCudaOp : Op {
            Var* output;
            MyCudaOp(NanoVector shape, string dtype="float");
            
            const char* name() const override { return "my_cuda"; }
            DECLARE_jit_run;
        };
        """, """
        #ifndef JIT
        MyCudaOp::MyCudaOp(NanoVector shape, string dtype) {
            flags.set(NodeFlags::_cuda);
            output = create_output(shape, dtype);
        }

        void MyCudaOp::jit_prepare() {
            add_jit_define("T", output->dtype());
        }

        #else // JIT
        #ifdef JIT_cuda

        __global__ void kernel(index_t n, T *x) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            int stride = blockDim.x * gridDim.x;
            for (int i = index; i < n; i += stride)
                x[i] = (T)-i;
        }

        void MyCudaOp::jit_run() {
            index_t num = output->num;
            auto* __restrict__ x = output->ptr<T>();
            int blockSize = 256;
            int numBlocks = (num + blockSize - 1) / blockSize;
            kernel<<<numBlocks, blockSize>>>(num, x);
        }
        #endif // JIT_cuda
        #endif // JIT
        """,
        "my_cuda")
        a = my_op([3,4,5], 'float')
        na = a.data
        assert a.shape == [3,4,5] and a.dtype == 'float'
        assert (-na.flatten() == range(3*4*5)).all(), na


@unittest.skipIf(jt.compiler.has_cuda, "Only test without CUDA")
class TestNoCuda(unittest.TestCase):
    def test_cuda_flags(self):
        expect_error(lambda: setattr(jt.flags, "use_cuda",1))

if __name__ == "__main__":
    unittest.main()
