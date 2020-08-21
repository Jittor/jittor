# ***************************************************************
# Copyright (c) 2020 Jittor. All Rights Reserved.
# Authors:
#   Dun Liang <randonlang@gmail.com>.
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jiyan as jy
import jittor as jt
import numpy as np

@jy.jit(backend=jy.numpy_backend)
def simple_add_2d(a, b):
    n, m = a.shape
    for i in range(n):
        for j in range(m):
            a[i,j] += b[i,j]

class TestNumpyBackend(unittest.TestCase):
    def test(self):
        a = jt.random((10,10))
        b = jt.random((10,10))
        prev_a = a.numpy()
        simple_add_2d(a, b)
        assert np.allclose(a.data, prev_a + b.data)

if __name__ == "__main__":
    unittest.main()
