# ***************************************************************
# Copyright (c) 2021 Jittor. All Rights Reserved. 
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


if __name__ == "__main__":
    unittest.main()