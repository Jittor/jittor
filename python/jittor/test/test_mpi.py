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

def main():
    jt.compile_extern.mpi_ops.mpi_test("").data == 123

@unittest.skipIf(jt.compile_extern.mpi_ops is None, "no mpi found")
class TestMpi(unittest.TestCase):
    def test(self):
        mpi = jt.compile_extern.mpi
        if mpi.world_size() == 1:
            mpirun_path = jt.compiler.env_or_try_find('mpirun_path', 'mpirun')
            cmd = f"{mpirun_path} -np 2 {sys.executable} -m jittor.test.test_mpi"
            print("run cmd", cmd)
            jt.compiler.run_cmd(cmd)
        else:
            main()

if __name__ == "__main__":
    unittest.main()