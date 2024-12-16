# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved. 
# Maintainers: Dun Liang <randonlang@gmail.com>. 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
import numpy as np

class TestGetItemSimple(unittest.TestCase):
    def test_get_by_pos_int(self):
        a = jt.array([-2,3,4,-5,-6])
        b = a[3]
        b.sync()
        assert b.item() == -5
    def test_get_by_neg_int(self):
        a = jt.array([-2,3,4,-5,-6])
        b = a[-3]
        b.sync()
        assert b.item() == 4
    def test_get_slice(self):
        a = jt.array([-2,3,4,-5,-6])
        b = a[-1:-3:-1].numpy().tolist()
        assert len(b) == 2
        assert b[0] == -6
        assert b[1] == -5
    def test_get_by_list(self):
        a = jt.array([-2,3,4,-5,-6])
        b = a[[-1, -3, 1]].numpy().tolist()
        assert len(b) == 3
        assert b[0] == -6
        assert b[1] == 4
        assert b[2] == 3
    def test_multidim_by_points(self):
        a = jt.arange(24).reshape(2, 3, 4)
        b = jt.array([0, 1, 0])
        c = jt.array([0, -1, 1])
        d = jt.array([-2, 0, 3])
        e = a[(b, c, d)].numpy().tolist()
        assert len(e) == 3
        assert e[0] == 2
        assert e[1] == 20
        assert e[2] == 7

if __name__ == "__main__":
    jt.flags.use_cuda = True
    unittest.main()