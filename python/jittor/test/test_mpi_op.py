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
from jittor.test.test_mpi import run_mpi_test

mpi = jt.compile_extern.mpi

@unittest.skipIf(mpi is None, "no inside mpirun")
class TestMpiOps(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        np.random.seed(0)
        jt.seed(3)

    def test_all_reduce(self):
        x = jt.random([5, 5])
        y = jt.compile_extern.mpi_ops.mpi_all_reduce(x)
        assert np.allclose(y.data, (x*3).data)
        g = jt.grad(y,x)
        assert np.allclose(g.data, np.ones([5,5])*3)

    def test_all_reduce_mean(self):
        x = jt.random([5, 5])
        y = jt.compile_extern.mpi_ops.mpi_all_reduce(x, "mean")
        assert np.allclose(y.data, x.data)
        g = jt.grad(y,x)
        assert np.allclose(g.data, np.ones([5,5]))

    def test_broadcast(self):
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

    def test_reduce(self):
        x = jt.random([5, 5])
        y = jt.compile_extern.mpi_ops.mpi_reduce(x, root=0)
        y.sync()
        if mpi.world_rank() == 0:
            assert np.allclose(y.data, (x*3).data)
        else:
            assert np.allclose(y.data, np.zeros([5,5]))
        g = jt.grad(y,x)
        assert np.allclose(g.data, np.ones([5,5]))


@unittest.skipIf(not jt.compile_extern.has_mpi, "no mpi found")
class TestMpiOpsEntry(unittest.TestCase):
    def test(self):
        run_mpi_test(3, "test_mpi_op")

if __name__ == "__main__":
    unittest.main()