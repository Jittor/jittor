# ***************************************************************
# Copyright (c) 2022 Jittor. All Rights Reserved. 
# Maintainers: 
#     Zheng-Ning Liu <lzhengning@gmail.com>
#     Dun Liang <randonlang@gmail.com>. 
# 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************


import unittest
import jittor as jt
import numpy as np


class TestRepeatOp(unittest.TestCase):
    def test_repeat(self):
        np_a = np.arange(5)
        jt_a = jt.array(np_a)

        np_b = np.tile(np_a, (2, 3))
        jt_b = jt.repeat(jt_a, (2, 3))

        assert np.allclose(np_b, jt_b.data)

        np_b = np.tile(np_a, (2, 3, 1))
        jt_b = jt.repeat(jt_a, (2, 3, 1))

        assert np.allclose(np_b, jt_b.data)

        np_a = np.arange(24).reshape(2, 3, 4)
        jt_a = jt.array(np_a)

        np_b = np.tile(np_a, (2, 3))
        jt_b = jt.repeat(jt_a, (2, 3))

        assert np.allclose(np_b, jt_b.data)


    def test_highdim(self):
        np_a = np.arange(64).reshape(2, 2, 2, 2, 2, 2)
        jt_a = jt.array(np_a)

        np_b = np.tile(np_a, (2, 3))
        jt_b = jt.repeat(jt_a, (2, 3))

        assert np.allclose(np_b, jt_b.data)

        np_b = np.tile(np_a, (2, 1, 1, 3))
        jt_b = jt.repeat(jt_a, (2, 1, 1, 3))

        assert np.allclose(np_b, jt_b.data)

        np_b = np.tile(np_a, (2, 1, 1, 1, 3, 1))
        jt_b = jt.repeat(jt_a, (2, 1, 1, 1, 3, 1))

        assert np.allclose(np_b, jt_b.data)

if __name__ == "__main__":
    unittest.main()