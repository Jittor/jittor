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
    mpi = jt.compile_extern.mpi
    x = jt.random([5, 5])
    y = jt.compile_extern.mpi_ops.mpi_all_reduce(x)
    assert np.allclose(y.data, (x*3).data)
    g = jt.grad(y,x)
    assert np.allclose(g.data, np.ones([5,5])*3)

def test_broadcast():
    print("test broadcast")
    mpi = jt.compile_extern.mpi
    data = jt.random([5, 5])
    if mpi.world_rank() == 0:
        x = data
    else:
        x = jt.zeros([5, 5])
    y = jt.compile_extern.mpi_ops.mpi_broadcast(x, 0)
    assert np.allclose(y.data, data.data)
    g = jt.grad(y,x)
    if mpi.world_rank() == 0:
        assert np.allclose(g.data, np.ones([5,5])*3)
    else:
        assert np.allclose(g.data, np.zeros([5,5]))

def test_reduce():
    print("test reduce")
    mpi = jt.compile_extern.mpi
    x = jt.random([5, 5])
    y = jt.compile_extern.mpi_ops.mpi_reduce(x, 0)
    y.sync()
    if mpi.world_rank() == 0:
        assert np.allclose(y.data, (x*3).data)
    else:
        assert np.allclose(y.data, np.zeros([5,5]))
    g = jt.grad(y,x)
    assert np.allclose(g.data, np.ones([5,5]))

def main():
    np.random.seed(0)
    jt.set_seed(3)
    with jt.flag_scope(use_cuda=0):
        if jt.compile_extern.mpi_ops:
            test_all_reduce()
            test_broadcast()
            test_reduce()

@unittest.skipIf(not jt.compile_extern.has_mpi, "no mpi found")
class TestMpiOps(unittest.TestCase):
    def test(self):
        mpi = jt.compile_extern.mpi
        if not jt.compile_extern.inside_mpi():
            mpirun_path = jt.compiler.env_or_try_find('mpirun_path', 'mpirun')
            cmd = f"{mpirun_path} -np 3 {sys.executable} -m jittor.test.test_mpi_op"
            print("run cmd", cmd)
            jt.compiler.run_cmd(cmd)
        else:
            main()

if __name__ == "__main__":
    unittest.main()