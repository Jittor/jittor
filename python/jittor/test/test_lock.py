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
from pathlib import Path

@unittest.skipIf(jt.compile_extern.mpi_ops is None, "no mpi found")
class TestLock(unittest.TestCase):
    def test(self):
        mpi = jt.compile_extern.mpi
        mpirun_path = jt.compiler.env_or_try_find('mpirun_path', 'mpirun')
        if os.environ.get('lock_full_test', '0') == '1':
            cache_path = os.path.join(str(Path.home()), ".cache", "jittor", "lock")
            cmd = f"rm -rf {cache_path} && cache_name=lock {mpirun_path} -np 2 {sys.executable} -m jittor.test.test_example"
        else:
            cache_path = os.path.join(str(Path.home()), ".cache", "jittor")
            cmd = f"{mpirun_path} -np 2 {sys.executable} -m jittor.test.test_example"
        print("run cmd", cmd)
        jt.compiler.run_cmd(cmd)


if __name__ == "__main__":
    unittest.main()