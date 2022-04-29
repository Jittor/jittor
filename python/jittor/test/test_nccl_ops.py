# ***************************************************************
# Copyright (c) 2022 Jittor. All Rights Reserved. 
# Maintainers: 
#     Guoye Yang <498731903@qq.com>
#     Guowei Yang <471184555@qq.com>
#     Dun Liang <randonlang@gmail.com>. 
# 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import os, sys
import jittor as jt
import numpy as np
from jittor import nn
from jittor import nn, Module
import copy
from jittor.test.test_log import find_log_with_re
from jittor.test.test_mpi import run_mpi_test
from jittor.compile_extern import mpi, nccl_ops
n = 2

@unittest.skipIf(nccl_ops is None, "nccl not found")
class TestNcclOps(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        np.random.seed(0)
        jt.seed(3)

    @jt.flag_scope(use_cuda=1)
    def test_all_reduce(self):
        with jt.log_capture_scope(enable_tuner=1, log_silent=1,
            log_v=1, log_vprefix="op.cc=100,exe=1000"
        ) as raw_log:
            x = jt.random([5, 5])
            y = x.mpi_all_reduce()
            assert np.allclose(y.data, (x*n).data)
            g = jt.grad(y,x)
            assert np.allclose(g.data, np.ones([5,5])*n)
        
        logs = find_log_with_re(raw_log, "(Jit op key (not )?found: nccl_all_reduce.*)")
        assert len(logs)==2, len(logs)

    @jt.flag_scope(use_cuda=1)
    def test_broadcast(self):
        with jt.log_capture_scope(enable_tuner=1, log_silent=1,
            log_v=1, log_vprefix="op.cc=100,exe=1000"
        ) as raw_log:
            data = jt.random([5, 5])
            if mpi.world_rank() == 0:
                x = data
            else:
                x = jt.zeros([5, 5])
            y = x.mpi_broadcast(0)
            assert np.allclose(y.data, data.data)
            g = jt.grad(y.sum(),x)
            g_ = g.data
            if mpi.world_rank() == 0:
                assert np.allclose(g_, np.ones([5,5])*n)
        logs = find_log_with_re(raw_log, "(Jit op key (not )?found: nccl_broadcast.*)")
        assert len(logs)==1, len(logs)

    @jt.flag_scope(use_cuda=1)
    def test_reduce(self):
        with jt.log_capture_scope(enable_tuner=1, log_silent=1,
            log_v=1, log_vprefix="op.cc=100,exe=1000"
        ) as raw_log:
            x = jt.random([5, 5])
            y = x.mpi_reduce(root=0)
            y_ = y.data
            x_ = (x*n).data
            if mpi.world_rank() == 0:
                assert np.allclose(y_, x_)
            g = jt.grad(y,x)
            assert np.allclose(g.data, np.ones([5,5]))
        logs = find_log_with_re(raw_log, "(Jit op key (not )?found: nccl_reduce.*)")
        assert len(logs)==1, len(logs)

    @jt.flag_scope(use_cuda=1)
    def test_sync(self):

        class Model(Module):
            def __init__(self):
                self.linear1 = nn.Linear(3, 3)
                self.linear2 = nn.Linear(3, 1024, False)

            def execute(self, x):
                x = self.linear1(x)
                x = nn.relu(x)
                return self.linear2(x)

        net = Model()
        if mpi.world_rank() == 0:
            net.linear1.weight *= 0
            net.linear2.weight *= 0
            net.linear1.bias *= 0
            net.linear1.weight += 1
            net.linear2.weight += 1
            net.linear1.bias += 1
        net.mpi_param_broadcast()
        assert np.allclose(net.linear1.weight.data, jt.ones(net.linear1.weight.shape).data)
        assert np.allclose(net.linear2.weight.data, jt.ones(net.linear2.weight.shape).data)
        assert np.allclose(net.linear1.bias.data, jt.ones(net.linear1.bias.shape).data)

    @jt.flag_scope(use_cuda=1)
    def test_optimizer(self):

        class Model2(Module):
            def __init__(self, input_size):
                self.linear1 = nn.Linear(input_size, 10)
                self.relu1 = nn.Relu()
                self.linear2 = nn.Linear(10, 1)
            def execute(self, x):
                x = self.linear1(x)
                x = self.relu1(x)
                return self.linear2(x)
                
        def get_data(n):
            for i in range(n):
                x = np.random.rand(50, 1)
                y = x*x
                yield jt.float32(x), jt.float32(y)

        num = 2000
        model = Model2(1)
        model.mpi_param_broadcast()
        optimizer = nn.SGD(model.parameters(), 0.1)
        dataset = list(enumerate(get_data(num)))
        for i in range(mpi.world_rank(), num, n):
            id, (x, y) = dataset[i]
            pred_y = model(x)
            loss = (pred_y - y)**2
            loss_mean = loss.mean()
            optimizer.step(loss_mean)
        assert loss_mean.data < 0.0025, loss_mean.data
        jt.clean()

@unittest.skipIf(not jt.compile_extern.has_mpi, "no mpi found")
class TestNcclOpsEntry(unittest.TestCase):
    def test(self):
        run_mpi_test(2, "test_nccl_ops")

if __name__ == "__main__":
    unittest.main()