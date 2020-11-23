# ***************************************************************
# Copyright (c) 2020 Jittor. Authors: 
#     Wenyang Zhou <576825820@qq.com>. 
# All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
import numpy as np
skip_this_test = False

@unittest.skipIf(skip_this_test, "No Torch found")
class TestSetitem(unittest.TestCase):
    def test_getitem(self):
        # test getitem for float32/float64/bool/int8/int32
        arr_float32 = jt.random((4,2,3))
        arr_float32_res = arr_float32[1:3,:,:]
        arr_float32_res.data[0,0,0] = 1
        assert arr_float32[1,0,0] == 1
        arr_float32_res.data[1,1,2] = 1
        assert arr_float32[2,1,2] == 1
        arr_float32[1,0,0] = 0
        # getitem and setitem do not conflict 
        assert arr_float32_res[0,0,0] == 1

        arr_bool = jt.bool(np.ones((4,2,3)))
        arr_bool_res = arr_bool[1:3,:,:]
        arr_bool_res.data[0,0,0] = False
        assert arr_bool[1,0,0] == False
        arr_bool_res.data[0,0,1] = False
        assert arr_bool[1,0,1] == False

        arr_float64 = jt.random((4,2,3), dtype='float64')
        arr_float64_res = arr_float64[1:3,:,:]
        arr_float64_res.data[0,0,0] = 1
        assert arr_float64[1,0,0] == 1
        arr_float64_res.data[1,1,2] = 1
        assert arr_float64[2,1,2] == 1

        arr_int32 = jt.ones((4,2,3), dtype='int32')
        arr_int32_res = arr_int32[1:3,:,:]
        arr_int32_res.data[0,0,0] = 0
        assert arr_int32[1,0,0] == 0
        arr_int32_res.data[1,1,2] = 0
        assert arr_int32[2,1,2] == 0
        
if __name__ == "__main__":
    unittest.main()