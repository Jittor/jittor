# ***************************************************************
# Copyright (c) 2021 Jittor. All Rights Reserved. 
# Maintainers: 
#    Dun Liang <randonlang@gmail.com>. 
# 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
import numpy as np

class TestMem(unittest.TestCase):
    def tearDown(self):
        jt.clean()
        jt.gc()

    @unittest.skipIf(not jt.has_cuda, "no cuda found")
    @jt.flag_scope(use_cuda=1)
    def test_oom(self):
        backups = []
        jt.flags.use_cuda = 1

        one_g = np.ones((1024*1024*1024//4,), "float32")

        meminfo = jt.get_mem_info()
        n = int(meminfo.total_cuda_ram // (1024**3) * 0.6)

        for i in range(n):
            a = jt.array(one_g)
            b = a + 1
            b.sync()
            backups.append((a,b))
        jt.sync_all(True)
        backups = []


if __name__ == "__main__":
    unittest.main()