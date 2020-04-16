# ***************************************************************
# Copyright (c) 2020 Jittor. Authors: 
#     Guoye Yang <498731903@qq.com>
#     Guowei Yang <471184555@qq.com>
#     Dun Liang <randonlang@gmail.com>. 
# All Rights Reserved.
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
n = 2
mpi = jt.compile_extern.mpi

def test_all_reduce():
    print("test all_reduce")
    x = jt.random([5, 5])
    y = jt.compile_extern.nccl_ops.nccl_all_reduce(x)
    assert np.allclose(y.data, (x*n).data)

def test_broadcast():
    print("test broadcast")
    data = jt.random([5, 5])
    if mpi.world_rank() == 0:
        x = data
    else:
        x = jt.zeros([5, 5])
    y = jt.compile_extern.nccl_ops.nccl_broadcast(x, 0)
    assert np.allclose(y.data, data.data)

def test_reduce():
    print("test reduce")
    x = jt.random([5, 5])
    y = jt.compile_extern.nccl_ops.nccl_reduce(x, 0)
    y_ = y.data
    x_ = (x*n).data
    if mpi.world_rank() == 0:
        assert np.allclose(y_, x_)

class Model(Module):
    def __init__(self):
        self.linear1 = nn.Linear(3, 3)
        self.linear2 = nn.Linear(3, 1024, False)

    def execute(self, x):
        x = self.linear1(x)
        x = nn.relu(x)
        return self.linear2(x)

def test_sync():
    print("test mpi_sync")
    net = Model()
    if mpi.world_rank() == 0:
        net.linear1.weight *= 0
        net.linear2.weight *= 0
        net.linear1.bias *= 0
        net.linear1.weight += 1
        net.linear2.weight += 1
        net.linear1.bias += 1
    net.mpi_sync()
    assert np.allclose(net.linear1.weight.data, jt.ones(net.linear1.weight.shape).data)
    assert np.allclose(net.linear2.weight.data, jt.ones(net.linear2.weight.shape).data)
    assert np.allclose(net.linear1.bias.data, jt.ones(net.linear1.bias.shape).data)

def main():
    np.random.seed(0)
    jt.set_seed(3)
    with jt.flag_scope(use_cuda=1):
        if jt.compile_extern.nccl_ops:
            test_sync()
            test_all_reduce()
            test_broadcast()
            test_reduce()

# @unittest.skipIf(not jt.compile_extern.has_mpi, "no mpi found")
# class TestNcclOps(unittest.TestCase):
#     def test(self):
#         mpi = jt.compile_extern.mpi
#         if mpi.world_size() == 1 and n != 1:
#             mpirun_path = jt.compiler.env_or_try_find('mpirun_path', 'mpirun')
#             cmd = f"{mpirun_path} -np {n} {sys.executable} -m jittor.test.test_nccl_ops"
#             print("run cmd", cmd)
#             jt.compiler.run_cmd(cmd)
#         else:
#             main()

@unittest.skipIf(mpi is None, "no inside mpirun")
class TestMpi(unittest.TestCase):
    def test(self):
        main()

@unittest.skipIf(not jt.compile_extern.has_mpi, "no mpi found")
class TestNcclOps(unittest.TestCase):
    def test_entry(self):
        if not jt.compile_extern.inside_mpi():
            mpirun_path = jt.compile_extern.mpicc_path.replace("mpicc", "mpirun")
            cmd = f"{mpirun_path} -np {n} {sys.executable} -m jittor.test.test_nccl_ops -v"
            print("run cmd:", cmd)
            assert os.system(cmd)==0, "run cmd failed: "+cmd

if __name__ == "__main__":
    unittest.main()