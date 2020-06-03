# ***************************************************************
# Copyright (c) 2020 Jittor. Authors: 
#     Wenyang Zhou <576825820@qq.com>
#     Dun Liang <randonlang@gmail.com>. 
# All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import os, sys
import jittor as jt
import numpy as np
mpi = jt.compile_extern.mpi

from jittor.dataset.mnist import MNIST
dataloader = MNIST(train=False).set_attrs(batch_size=16)

def val1():
    for i, (imgs, labels) in enumerate(dataloader):
        assert(imgs.shape[0]==8)
        if i == 5:
            break

@jt.single_process_scope(rank=0)
def val2():
    for i, (imgs, labels) in enumerate(dataloader):
        assert(imgs.shape[0]==16)
        if i == 5:
            break

@unittest.skipIf(mpi is None, "no inside mpirun")
class TestSingleProcessScope(unittest.TestCase):
    def test_single_process_scope(self):
        val1()
        val2()

def run_single_process_scope_test(num_procs, name):
    if not jt.compile_extern.inside_mpi():
        mpirun_path = jt.compile_extern.mpicc_path.replace("mpicc", "mpirun")
        cmd = f"{mpirun_path} -np {num_procs} {sys.executable} -m jittor.test.{name} -v"
        print("run cmd:", cmd)
        assert os.system(cmd)==0, "run cmd failed: "+cmd

@unittest.skipIf(not jt.compile_extern.has_mpi, "no mpi found")
class TestSingleProcessScopeEntry(unittest.TestCase):
    def test_entry(self):
        run_single_process_scope_test(2, "test_single_process_scope")

if __name__ == "__main__":
    unittest.main()