# ***************************************************************
# Copyright (c) 2020 Jittor. Authors: 
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

def test_all_reduce():
    print("test all_reduce")
    x = jt.random([5, 5])
    y = jt.compile_extern.nccl_ops.nccl_all_reduce(x)
    assert np.allclose(y.data, (x*3).data)

def test_broadcast():
    print("test broadcast")
    mpi = jt.compile_extern.mpi
    data = jt.random([5, 5])
    if mpi.world_rank() == 0:
        x = data
    else:
        x = jt.zeros([5, 5])
    y = jt.compile_extern.nccl_ops.nccl_broadcast(x, 0)
    assert np.allclose(y.data, data.data)

def test_reduce():
    print("test reduce")
    mpi = jt.compile_extern.mpi
    x = jt.random([5, 5])
    y = jt.compile_extern.nccl_ops.nccl_reduce(x, 0)
    y.sync()
    if mpi.world_rank() == 0:
        assert np.allclose(y.data, (x*3).data)

def main():
    np.random.seed(0)
    jt.set_seed(3)
    with jt.flag_scope(use_cuda=1):
        if jt.compile_extern.nccl_ops:
            test_all_reduce()
            test_broadcast()
            test_reduce()

@unittest.skipIf(jt.compile_extern.mpi_ops is None, "no mpi found")
class TestNcclOps(unittest.TestCase):
    def test(self):
        mpi = jt.compile_extern.mpi
        if mpi.world_size() == 1:
            mpirun_path = jt.compiler.env_or_try_find('mpirun_path', 'mpirun')
            cmd = f"{mpirun_path} -np 3 {sys.executable} -m jittor.test.test_nccl_ops"
            print("run cmd", cmd)
            jt.compiler.run_cmd(cmd)
        else:
            main()

if __name__ == "__main__":
    unittest.main()