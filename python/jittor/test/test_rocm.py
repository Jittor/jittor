# ***************************************************************
# Copyright (c) 2021 Jittor. All Rights Reserved. 
# Maintainers: Zheng-Ning Liu <lzhengning@gmail.com>. 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest

import os
import random
import math
import time

import numpy as np
import tqdm

import jittor as jt
from jittor import init, Module, nn, Function
from jittor.models import vgg
from jittor.dataset.mnist import MNIST
import jittor.transform as trans

from .test_core import expect_error
from .test_reorder_tuner import simple_parser
from .test_log import find_log_with_re


def test_rocm(use_rocm=1):
    @unittest.skipIf(not jt.compiler.has_rocm, "No ROCm found")
    class TestCudaBase(unittest.TestCase):
        def setUp(self):
            jt.flags.use_rocm = use_rocm
        def tearDown(self):
            jt.flags.use_rocm = 0
    return TestCudaBase


@unittest.skipIf(not jt.compiler.has_rocm, "No ROCm found")
class TestROCm(unittest.TestCase):

    @jt.flag_scope(use_rocm=1)
    def test_array(self):
        a = jt.array([1,2,3])
        np.testing.assert_allclose(a.numpy(), [1,2,3])

    @jt.flag_scope(use_rocm=1)
    def test_add(self):
        a = jt.array([1,2,3])
        b = a+a
        np.testing.assert_allclose(b.numpy(), [2,4,6])

    @jt.flag_scope(use_rocm=1)
    def test_add_float(self):
        a = jt.array([1.0,2.0,3.0])
        b = a+a
        np.testing.assert_allclose(b.numpy(), [2,4,6])

    @jt.flag_scope(use_rocm=1)
    def test_array_cast(self):
        # this test cannot pass because cast error
        x = np.random.rand(10)
        y = jt.float32(x)
        np.testing.assert_allclose(x, y.numpy())

    def test_meminfo(self):
        jt.display_memory_info()

    @jt.flag_scope(use_rocm=1)
    def test_cuda_flags(self):
        a = jt.random((10, 10))
        a.sync()

    @jt.flag_scope(use_rocm=1)
    def test_rocm_custom_op_from_cuda(self):
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
        na = a.data
        assert a.shape == [3,4,5] and a.dtype == 'float'
        assert (-na.flatten() == range(3*4*5)).all(), na

    def test_rocm_fused_op(self):
        a = jt.array([1,2,3])
        a.sync()
        with jt.flag_scope(use_rocm=1):
            ((a+a)*2).data


class Model(Module):
    def __init__(self, input_size):
        self.linear1 = nn.Linear(input_size, 10)
        self.relu1 = nn.Relu()
        self.linear2 = nn.Linear(10, 1)
    def execute(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        return self.linear2(x)


@unittest.skipIf(not jt.compiler.has_rocm, "No ROCm found")
class TestExample(unittest.TestCase):
    @jt.flag_scope(use_rocm=1)
    def test1(self):
        np.random.seed(0)
        jt.set_seed(3)
        n = 1000
        batch_size = 50
        lr = 0.05

        def get_data(n):
            for i in range(n):
                x = np.random.rand(batch_size, 1).astype("float32")
                y = x*x
                yield jt.float32(x), jt.float32(y)
        
        model = Model(input_size=1)
        ps = model.parameters()

        for i,(x,y) in enumerate(get_data(n)):
            jt.sync_all(True)
            pred_y = model(x).name("pred_y")
            loss = ((pred_y - y).sqr()).name("loss")
            loss_mean = loss.mean()
            
            gs = jt.grad(loss_mean, ps)
            for p, g in zip(ps, gs):
                p -= g * lr

            if i>2:
                assert prev == jt.liveness_info(), f"memory leak {prev} {jt.liveness_info()}"
            prev = jt.liveness_info()

        possible_results = [
            0.0009948202641680837,
            0.001381353591568768,
            0.00110957445576787,
            0.001124994712881744
        ]
        loss_mean = loss_mean.data
        assert any(abs(loss_mean - r) < 1e-6 for r in possible_results)

        jt.clean()


from .test_unary_op import TestUnaryOp
@unittest.skipIf(not jt.compiler.has_rocm, "No ROCm found")
class TestROCmUnaryOp(TestUnaryOp, test_rocm(1)):
    pass


from .test_binary_op import TestBinaryOp
@unittest.skipIf(not jt.compiler.has_rocm, "No ROCm found")
class TestROCmBinaryOp(TestBinaryOp, test_rocm(1)):
    pass


from .test_reduce_op import TestReduceOp
@unittest.skipIf(not jt.compiler.has_rocm, "No ROCm found")
class TestROCmReduceOp(TestReduceOp, test_rocm(1)):
    pass


from .test_reindex_op import TestReindexOp
@unittest.skipIf(not jt.compiler.has_rocm, "No ROCm found")
class TestROCmReindexOp(TestReindexOp, test_rocm(1)):
    pass


from .test_where_op import TestWhereOp
@unittest.skipIf(not jt.compiler.has_rocm, "No ROCm found")
class TestROCmWhereOp(TestWhereOp, test_rocm(1)):
    pass


# from .test_reindex_reduce_op import TestReindexReduceOp
# @unittest.skipIf(not jt.compiler.has_rocm, "No ROCm found")
# class TestROCmReindexReduceOp(TestReindexReduceOp, test_rocm(1)):
#     pass


@unittest.skipIf(not jt.compiler.has_rocm, "No ROCm found")
class TestROCmCodeOp(unittest.TestCase):
    @jt.flag_scope(use_rocm=1)
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

    @jt.flag_scope(use_rocm=1)
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

    @jt.flag_scope(use_rocm=1)
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


@unittest.skipIf(not jt.compiler.has_rocm, "No ROCm found")
class TestBMM(unittest.TestCase):
    def test_bmm_rocm(self):
        def check(batch, n, m, k):
            def calc(use_rocm, a, b, mask):
                jt.flags.use_rocm = use_rocm
                a = jt.array(a)
                b = jt.array(b)
                mask = jt.array(mask)
                c = nn.bmm(a, b)
                da, db = jt.grad(c*mask, [a, b])
                return c.data, da.data, db.data
            mask = np.random.rand(batch, n, k).astype("float32")
            a = np.random.rand(batch, n, m).astype("float32")
            b = np.random.rand(batch, m, k).astype("float32")
            a1,a2,a3 = calc(0, a, b, mask)
            b1,b2,b3 = calc(1, a, b, mask)
            assert np.allclose(a1, b1)
            assert np.allclose(a2, b2)
            assert np.allclose(a3, b3)
        check(10,3,4,5)
        check(10,8,8,8)
        check(10,8,1,8)
        check(10,8,8,1)
        check(10,1,8,8)
        check(1,7,8,8)

class Model(Module):
    def __init__(self, input_size):
        self.linear1 = nn.Linear(input_size, 10)
        self.relu1 = nn.Relu()
        self.linear2 = nn.Linear(10, 1)
    def execute(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        return self.linear2(x)

from jittor.models import resnet

class MnistNet(Module):
    def __init__(self):
        self.model = resnet.Resnet18()
        self.layer = nn.Linear(1000,10)
    def execute(self, x):
        x = self.model(x)
        x = self.layer(x)
        return x

@unittest.skipIf(not jt.compiler.has_rocm, "skip_this_test")
class TestResnetFp32(unittest.TestCase):
    # setup random seed
    def setup_seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)
        jt.seed(seed)

    @jt.flag_scope(use_cuda=1)
    def test_resnet(self):
        self.setup_seed(1)

        # hyper-parameters
        self.batch_size = int(os.environ.get("TEST_BATCH_SIZE", "100"))
        self.weight_decay = 0.0001
        self.momentum = 0.9
        self.learning_rate = 0.1
        if jt.flags.amp_reg:
            self.learning_rate = 0.01
        # mnist dataset
        self.train_loader = MNIST(train=True, transform=trans.Resize(224)) \
            .set_attrs(batch_size=self.batch_size, shuffle=True)
        self.train_loader.num_workers = 4

        loss_list=[]
        acc_list=[]
        mnist_net = MnistNet()
        global prev
        SGD = nn.SGD(mnist_net.parameters(), self.learning_rate, self.momentum, self.weight_decay)
        self.train_loader.endless = True

        for data, target in self.train_loader:
            batch_id = self.train_loader.batch_id
            epoch_id = self.train_loader.epoch_id
            data = data.float_auto()
            output = mnist_net(data)
            loss = nn.cross_entropy_loss(output, target)

            break
        jt.sync_all(True)
        
        for _ in range(10):
            output = mnist_net(data)
            loss = nn.cross_entropy_loss(output, target)
            SGD.step(loss)
            def callback(epoch_id, batch_id, loss, output, target):
                pred = np.argmax(output, axis=1)
                acc = np.mean(target==pred)
            jt.fetch(epoch_id, _, loss, output, target, callback)
        jt.sync_all(True)

        all_time = time.time()
        prev = time.time()
        print('starting')
        for _ in range(100):
            output = mnist_net(data)
            loss = nn.cross_entropy_loss(output, target)
            SGD.step(loss)
            def callback(epoch_id, batch_id, loss, output, target):
                global prev
                pred = np.argmax(output, axis=1)
                acc = np.mean(target==pred)
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {:.6f} \tTime:{:.3f}'
                    .format(epoch_id, batch_id, 600,1. * batch_id / 6.0, loss[0], acc, time.time()-prev))
                prev = time.time()
            jt.fetch(epoch_id, _, loss, output, target, callback)
        jt.sync_all(True)
        print(f'all = {time.time() - all_time}')



if __name__ == "__main__":
    unittest.main()