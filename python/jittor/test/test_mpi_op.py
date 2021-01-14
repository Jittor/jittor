# ***************************************************************
# Copyright (c) 2021 Jittor. All Rights Reserved. 
# Maintainers: 
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
from jittor.test.test_mpi import run_mpi_test

mpi = jt.compile_extern.mpi
if mpi:
    n = mpi.world_size()

@unittest.skipIf(not jt.in_mpi, "no inside mpirun")
class TestMpiOps(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        np.random.seed(0)
        jt.seed(3)

    def test_all_reduce(self):
        x = jt.random([5, 5])
        y = x.mpi_all_reduce()
        assert np.allclose(y.data, (x*n).data)
        g = jt.grad(y,x)
        assert np.allclose(g.data, np.ones([5,5])*n)

    def test_all_reduce_mean(self):
        x = jt.random([5, 5])
        y = x.mpi_all_reduce("mean")
        assert np.allclose(y.data, x.data)
        g = jt.grad(y,x)
        assert np.allclose(g.data, np.ones([5,5]))

    def test_broadcast(self):
        data = jt.random([5, 5])
        if mpi.world_rank() == 0:
            x = data
        else:
            x = jt.zeros([5, 5])
        y = x.mpi_broadcast(0)
        assert np.allclose(y.data, data.data)
        g = jt.grad(y,x)
        if mpi.world_rank() == 0:
            assert np.allclose(g.data, np.ones([5,5])*n)
        else:
            assert np.allclose(g.data, np.zeros([5,5]))

    def test_reduce(self):
        x = jt.random([5, 5])
        y = x.mpi_reduce(root=0)
        y.sync()
        if mpi.world_rank() == 0:
            assert np.allclose(y.data, (x*n).data)
        else:
            assert np.allclose(y.data, np.zeros([5,5]))
        g = jt.grad(y,x)
        assert np.allclose(g.data, np.ones([5,5]))


@unittest.skipIf(not jt.compile_extern.has_mpi, "no mpi found")
class TestMpiOpsEntry(unittest.TestCase):
    def test(self):
        run_mpi_test(2, "test_mpi_op")

if __name__ == "__main__":
    unittest.main()