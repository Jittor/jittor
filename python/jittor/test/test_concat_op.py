# ***************************************************************
# Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
import numpy as np


class TestConcatOp(unittest.TestCase):
    def test_concat_op(self):
        def check(tmp, dim=0):
            res1 = jt.WIP_concat(tmp, dim=dim)
            res2 = jt.contrib.concat(tmp, dim=dim)
            assert (res1!=res2).data.sum()==0, "concat fail..."
        check([jt.array([[1],[2]]), jt.array([[2],[2]])])
        check([jt.array(np.array(range(24))).reshape((1,2,3,4)), jt.array(np.array(range(24))).reshape((1,2,3,4))])
        check([jt.array(np.array(range(120))).reshape((5,2,3,4)), jt.array(np.array(range(24))).reshape((1,2,3,4))])
        check([jt.array(np.array(range(5))).reshape((5,1)), jt.array(np.array(range(1))).reshape((1,1))])
        print('concat success...')  

if __name__ == "__main__":
    unittest.main()