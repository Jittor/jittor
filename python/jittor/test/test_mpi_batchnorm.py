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

def test_batchnorm():
    print("test batchnorm")
    mpi = jt.compile_extern.mpi
    data = np.random.rand(30,3,10,10)
    x1 = jt.array(data)
    x2 = jt.array(data[mpi.world_rank()*10:(mpi.world_rank()+1)*10,...])
    
    bn1 = nn.BatchNorm(3)
    bn2 = nn.SyncBatchNorm(3)
    y1 = bn1(x1).data
    y2 = bn2(x2).data

    assert bn1.running_mean==bn2.running_mean
    assert bn1.running_var==bn2.running_var

def main():
    np.random.seed(0)
    jt.set_seed(3)
    with jt.flag_scope(use_cuda=0):
        test_batchnorm()
    with jt.flag_scope(use_cuda=1):
        test_batchnorm()

@unittest.skipIf(not jt.compile_extern.has_mpi, "no mpi found")
class TestMpiOps(unittest.TestCase):
    def test(self):
        mpi = jt.compile_extern.mpi
        if not jt.compile_extern.inside_mpi():
            mpirun_path = jt.compiler.env_or_try_find('mpirun_path', 'mpirun')
            cmd = f"{mpirun_path} -np 3 {sys.executable} -m jittor.test.test_mpi_batchnorm"
            print("run cmd", cmd)
            jt.compiler.run_cmd(cmd)
        else:
            main()

if __name__ == "__main__":
    unittest.main()