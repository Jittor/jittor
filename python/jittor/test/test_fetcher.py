# ***************************************************************
# Copyright (c) 2022 Jittor. All Rights Reserved. 
# Maintainers: Dun Liang <randonlang@gmail.com>. 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
import numpy as np
from jittor import compile_extern

class TestFetcher(unittest.TestCase):
    def test_fetch(self):
        a = jt.array([1,2,3])
        a = a*2
        v = []
        jt.fetch(a, lambda a: v.append(a))
        jt.fetch(1, 2, 3, a, 
            lambda x, y, z, a: self.assertTrue(x==1 and y==2 and z==3 and isinstance(a, np.ndarray))
        )
        jt.sync_all(True)
        assert len(v)==1 and (v[0]==[2,4,6]).all()

@unittest.skipIf(not jt.has_cuda, "Cuda not found")
class TestFetcherCuda(TestFetcher):
    @classmethod
    def setUpClass(self):
        jt.flags.use_cuda = 1

    @classmethod
    def tearDownClass(self):
        jt.flags.use_cuda = 0

if __name__ == "__main__":
    unittest.main()