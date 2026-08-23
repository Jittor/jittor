# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved. 
# Maintainers: Dun Liang <randonlang@gmail.com>. 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
import numpy as np
from _helpers.assertions import expect_error

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

        void NoCudaOp::jit_prepare(JK& jk) {
            add_jit_define(jk, "T", output->dtype());
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

        void MyCudaOp::jit_prepare(JK& jk) {
            add_jit_define(jk, "T", output->dtype());
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
        na = a.numpy()
        assert a.shape == [3,4,5] and str(a.dtype) == "float32"
        assert (-na.flatten() == range(3*4*5)).all(), na

    @jt.flag_scope(use_cuda=2)
    def test_forced_cuda_fused_scalar_array(self):
        value = np.array([0.25, 0.5, 0.75], dtype="float32")
        a = jt.array(value)
        out = (a.exp() + 1).numpy()
        grad = jt.grad(jt.abs(a), a).numpy()
        np.testing.assert_allclose(out, np.exp(value) + 1, rtol=1e-6, atol=1e-6)
        np.testing.assert_array_equal(grad, np.ones_like(value))

    def test_cuda_fused_op(self):
        a = jt.array([1,2,3])
        a.sync()
        with jt.flag_scope(use_cuda=1):
            ((a+a)*2).data

    @jt.flag_scope(use_cuda=1)
    def test_large_nchw_channel_bias_broadcast(self):
        rng = np.random.RandomState(20260822)
        for shape in ((2, 32, 16, 16), (2, 64, 32, 32)):
            value = rng.randn(*shape).astype("float32")
            bias = rng.randn(shape[1]).astype("float32")
            actual = (
                jt.array(value)
                + jt.array(bias).broadcast(shape, [0, 2, 3])
            ).numpy()
            expected = value + bias.reshape(1, -1, 1, 1)
            np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


@unittest.skipIf(jt.compiler.has_cuda, "Only test without CUDA")
class TestNoCuda(unittest.TestCase):
    def test_cuda_flags(self):
        expect_error(lambda: setattr(jt.flags, "use_cuda",1))

if __name__ == "__main__":
    unittest.main()
