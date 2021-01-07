# ***************************************************************
# Copyright (c) 2021 Jittor. All Rights Reserved. 
# Maintainers: 
#     Wenyang Zhou <576825820@qq.com>. 
# 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
import numpy as np
skip_this_test = False

@unittest.skipIf(skip_this_test, "No Torch found")
class TestSetitem(unittest.TestCase):
    def test_setitem(self):
        arr0 = jt.random((4,2,2))
        data0 = jt.ones((2,2))
        arr0[1] = data0
        arr0.sync()
        data0.data[0,0] = 0
        assert arr0[1,0,0] == 0

        arr00 = jt.random((4,2,2))
        data00 = jt.ones((2,2))
        # share memory will fail if d has an edge to other nodes.
        tmp = data00 + 1
        arr00[1] = data00
        arr00.sync()
        data00.data[0,0] = 0
        assert arr00[1,0,0] == 0

        arr1 = jt.random((4,2,2))
        data1 = jt.zeros((2,2))
        arr1[3,:,0:2] = data1
        arr1.sync()
        data1.data[0,0] = 1
        assert arr1[3,0,0] == 1

        arr21 = jt.ones((2,2))
        arr22 = jt.ones((2,2)) * 2
        arr2 = jt.contrib.concat([arr21, arr22], dim=0)
        arr2.sync()
        arr21.data[0,0] = 3
        arr22.data[0,0] = 4
        assert arr2[0,0] == 3
        assert arr2[2,0] == 4

    def test_getitem(self):
        # test for different slice type
        arr0 = jt.random((4,3))
        arr0_res = arr0[2,:]
        arr0_res.data[1] = 1
        assert arr0[2,1] == 1

        arr1 = jt.array([1,2,3,4])
        arr1_res = arr1[None]
        arr1_res.data[0,2] = -1
        assert arr1[2] == -1

        arr2 = jt.array([1,2,3,4])
        arr2_res = arr2[...]
        arr2_res.data[2] = -1
        assert arr2[2] == -1

        arr3 = jt.array([1,2,3,4])
        arr3_res = arr3[3]
        arr3_res.data[0] = -1
        assert arr3[3] == -1

        arr4 = jt.random((4,2,3,3))
        arr4_res = arr4[...,:,:]
        arr4_res.data[0,0,1,1] = 1
        assert arr4[0,0,1,1] == 1

        arr5 = jt.random((4,2,3,3))
        arr5_res = arr5[1:3,:,:,:]
        arr5_res.data[1,0,1,1] = 1
        assert arr5[2,0,1,1] == 1

        arr6 = jt.random((4,2,3,3))
        arr6_res = arr6[1]
        arr6_res.data[0,1,1] = 1
        assert arr6[1,0,1,1] == 1

        # test for different data type (float32/float64/bool/int8/int32)
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

    def test_setitem_inplace_case1(self):
        # test type case
        a = jt.zeros((3,))
        a[1] = 123
        assert a.data[1] == 123

    def test_setitem_inplace_case2(self):
        # test un-continuous first dim
        a = jt.zeros((3,))
        a[0::2] = jt.ones((2,))
        assert a.data[2] == 1

    def test_setitem_inplace_case3(self):
        # test broadcast
        a = jt.zeros((3,))
        a[0:] = 1.0
        assert a.data[2] == 1
        
if __name__ == "__main__":
    unittest.main()