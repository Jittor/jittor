# ***************************************************************
# Copyright (c) 2022 Jittor. All Rights Reserved. 
# Maintainers: Dun Liang <randonlang@gmail.com>. 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
import numpy as np
from .test_core import expect_error

class TestContrib(unittest.TestCase):
    def test_concat(self):
        def check(shape, dim, n):
            num = np.prod(shape)
            arr1 = []
            arr2 = []
            for i in range(n):
                a = (np.array(range(num)) + i*num).reshape(shape)
                arr1.append(a)
                arr2.append(jt.array(a))
            x = np.concatenate(tuple(arr1), dim)
            y = jt.concat(arr2, dim)
            assert (x==y.data).all(), (x, y.data, arr1, arr2)
        check([2,3,4], 0, 2)
        check([2,3,4], 1, 3)
        check([2,3,4], 2, 4)
        check([2,3,4,5], 0, 4)
        check([2,3,4,5], 2, 4)
        check([2,3,4,5], 3, 4)
        check([1], 0, 20)

    def test_slice(self):
        def check(shape, slices):
            x = jt.random(shape)
            a = x[slices].data
            b = x.data[slices]
            assert (a==b).all(), (a, b)
            y = x.numpy()
            v = jt.random(a.shape)
            x[slices] = v
            y[slices] = v.data
            assert (x.data==y).all()
        # TODO: when slice same row/col many times and assign value, numpy will retain the last value but we assign their sum. eg: check([3,3,3,3], ([[0,1,1]],slice(None),[[1],[2],[0]],1))
        check([3], (1))
        check([3,3,3,3], ([[0],[1]],slice(None),[1,2],1))
        check([3,3,3,3], (slice(None),slice(None),slice(None),slice(None)))
        check([3,3,3,3], ([0,1],[0,1],[0,1],[0,1]))
        check([3,3,3,3], ([0,1],-2,slice(None),[0,1]))
        check([3,3,3,3], ([0,1],slice(1,2,2),[1,2],1))
        check([3,3,3,3], ([0,1],slice(None),[1,2],1))
        check([10,10,10,10], (slice(1,None,2),slice(-1,None,2),[1,2],-4))
        check([20], 0)
        check([20], 10)
        check([20], -10)
        check([10,10,10,10], 1)
        check([10,10,10,10], (1,slice(None),2))
        check([10,10,10,10], (-2,slice(None),2,slice(1,9,2)))


if __name__ == "__main__":
    unittest.main()
