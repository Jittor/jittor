# ***************************************************************
# Copyright (c) 2020 Jittor.
# Authors:
#     Dun Liang <randonlang@gmail.com>.
# All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
import numpy as np


class TestSlice(unittest.TestCase):
    def test_slice_bool(self):
        a = jt.zeros(10, "bool")
        a[1] = True
        a[2] = 1
        assert a.dtype == "bool"
        print(a)


if __name__ == "__main__":
    unittest.main()
