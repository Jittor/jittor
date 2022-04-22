# ***************************************************************
# Copyright (c) 2022 Jittor. All Rights Reserved. 
# Maintainers: 
#     Wenyang Zhou <576825820@qq.com>
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

from jittor.dataset.mnist import MNIST

def val1():
    dataloader = MNIST(train=False).set_attrs(batch_size=16)
    for i, (imgs, labels) in enumerate(dataloader):
        assert(imgs.shape[0]==8)
        if i == 5:
            break

@jt.single_process_scope(rank=0)
def val2():
    dataloader = MNIST(train=False).set_attrs(batch_size=16)
    for i, (imgs, labels) in enumerate(dataloader):
        assert(imgs.shape[0]==16)
        if i == 5:
            break

@unittest.skipIf(not jt.in_mpi, "no inside mpirun")
class TestSingleProcessScope(unittest.TestCase):
    def test_single_process_scope(self):
        val1()
        val2()

@unittest.skipIf(not jt.compile_extern.has_mpi, "no mpi found")
class TestSingleProcessScopeEntry(unittest.TestCase):
    def test_entry(self):
        run_mpi_test(2, "test_single_process_scope")

if __name__ == "__main__":
    unittest.main()