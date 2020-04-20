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
from jittor import nn
import numpy as np
from jittor.test.test_mpi import run_mpi_test

mpi = jt.compile_extern.mpi

@unittest.skipIf(mpi is None, "no inside mpirun")
class TestMpiBatchnorm(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        np.random.seed(0)
        jt.seed(3)

    def test_batchnorm(self):
        mpi = jt.compile_extern.mpi
        data = np.random.rand(30,3,10,10).astype("float32")
        x1 = jt.array(data)
        x2 = jt.array(data[mpi.world_rank()*10:(mpi.world_rank()+1)*10,...])
        
        bn1 = nn.BatchNorm(3, sync=True)
        bn2 = nn.BatchNorm(3, sync=False)
        y1 = bn1(x1).data
        y2 = bn2(x2).data

        assert np.allclose(bn1.running_mean.data, bn2.running_mean.data), \
            (bn1.running_mean.data, bn2.running_mean.data)
        assert np.allclose(bn1.running_var.data, bn2.running_var.data)

    @unittest.skipIf(not jt.has_cuda, "no cuda")
    @jt.flag_scope(use_cuda=1)
    def test_batchnorm_cuda(self):
        self.test_batchnorm()


@unittest.skipIf(not jt.compile_extern.has_mpi, "no mpi found")
class TestMpiBatchnormEntry(unittest.TestCase):
    def test(self):
        run_mpi_test(3, "test_mpi_batchnorm")

if __name__ == "__main__":
    unittest.main()
