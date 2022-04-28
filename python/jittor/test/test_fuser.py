# ***************************************************************
# Copyright (c) 2022 Jittor. All Rights Reserved. 
# Maintainers: Dun Liang <randonlang@gmail.com>. 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
import numpy as np


class TestFuser(unittest.TestCase):
    def test_wrong_fuse(self):
        a = jt.array([1])
        b = jt.random([10,])
        c = (a * b).sum() + (a + 1)
        print(c)

    def test_wrong_fuse2(self):
        a = jt.array([1])
        b = jt.random([10,])
        c = jt.random([100,])
        bb = a*b
        cc = a*c
        jt.sync([bb,cc])
        np.testing.assert_allclose(b.data, bb.data)
        np.testing.assert_allclose(c.data, cc.data)

    def test_for_fuse(self):
        arr = []
        x = 0
        for i in range(100):
            arr.append(jt.array(1))
            x += arr[-1]
        x.sync()
        for i in range(100):
            # print(arr[i].debug_msg())
            assert ",0)" not in arr[i].debug_msg()

    def test_array_bc(self):
        # a = jt.array(1)
        with jt.profile_scope() as rep:
            b = jt.array(1).broadcast([10])
            b.sync()
        assert len(rep) == 2


if __name__ == "__main__":
    unittest.main()