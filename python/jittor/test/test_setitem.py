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
from jittor.test.test_log import find_log_with_re
skip_this_test = False

@unittest.skipIf(skip_this_test, "No Torch found")
class TestSetitem(unittest.TestCase):
    def test_getitem_grad_opt1(self):
        with jt.log_capture_scope(log_vprefix="setitem_gopt=1000") as rep:
            with jt.flag_scope(trace_py_var=2):
                v = jt.random((1,2,3,4,5,6))
                for i in range(6):
                    loss = 0.
                    ss = 1
                    if v.shape[i] % 2 == 0: 
                        ss = 2
                    res = v.split(split_size=ss, dim=i)
                    t_ = []
                    for j in range(len(res)):
                        t = np.random.random(res[j].shape).astype("float32")
                        loss += res[j] * t
                        t_.append(t)
                    dv = jt.grad(loss, v)
                    dv.sync()
                    data = jt.dump_trace_data()
                    jt.clear_trace_data()
                    assert (dv.numpy() == np.concatenate(t_, i)).all()
        logs = find_log_with_re(rep, "getitem_grad_opt happens")
        assert len(logs) == 4

    def test_setitem_grad_opt1(self):
        with jt.log_capture_scope(log_vprefix="setitem_gopt=1000") as rep:
            v = jt.random((4,10,10))
            a = jt.ones((10,10))
            b = jt.ones((10,10))
            c = jt.ones((10,10))
            d = jt.ones((10,10))
            v[0] = a
            v[1] = b
            v[2] = c
            v[3] = d
            t = np.random.random((4,10,10)).astype("float32")
            loss = v*t
            da, db, dc, dd = jt.grad(loss, [a,b,c,d])
            jt.sync([da, db, dc, dd])
            assert (da.numpy() == t[0]).all()
            assert (db.numpy() == t[1]).all()
            assert (dc.numpy() == t[2]).all()
            assert (dd.numpy() == t[3]).all()
        logs = find_log_with_re(rep, "setitem_grad_opt happens")
        logs1 = find_log_with_re(rep, "setitem_grad_opt set success")
        assert len(logs) == 1
        assert len(logs1) == 3
    
    def test_setitem_grad_opt2(self):
        with jt.log_capture_scope(log_vprefix="setitem_gopt=1000") as rep:
            v = jt.random((4,10,10))
            a = jt.ones((10,10))
            b = jt.ones((10,10))
            c = jt.ones((10,10))
            d = jt.ones((10,10))
            v[0] = a
            v[2] = c
            v[1] = b
            v[3] = d
            t = np.random.random((4,10,10)).astype("float32")
            loss = v*t
            da, db, dc, dd = jt.grad(loss, [a,b,c,d])
            jt.sync([da, db, dc, dd])
            assert (da.numpy() == t[0]).all()
            assert (db.numpy() == t[1]).all()
            assert (dc.numpy() == t[2]).all()
            assert (dd.numpy() == t[3]).all()
        logs = find_log_with_re(rep, "setitem_grad_opt happens")
        logs1 = find_log_with_re(rep, "setitem_grad_opt set success")
        assert len(logs) == 1
        assert len(logs1) == 2
    
    def test_setitem_grad_opt3(self):
        with jt.log_capture_scope(log_vprefix="setitem_gopt=1000") as rep:
            a = jt.ones((10,10,10))
            b = jt.ones((10,10,10))
            c = jt.ones((10,10,10))
            d = jt.ones((10,10,10))
            v = jt.contrib.concat([a,b,c,d], dim=1)
            t = np.random.random((10,40,10)).astype("float32")
            loss = v*t
            da, db, dc, dd = jt.grad(loss, [a,b,c,d])
            jt.sync([da, db, dc, dd])
            assert (da.numpy() == t[:,:10]).all()
            assert (db.numpy() == t[:,10:20]).all()
            assert (dc.numpy() == t[:,20:30]).all()
            assert (dd.numpy() == t[:,30:40]).all()
        logs = find_log_with_re(rep, "setitem_grad_opt happens")
        logs1 = find_log_with_re(rep, "setitem_grad_opt set success")
        assert len(logs) == 1
        assert len(logs1) == 3

    def test_setitem1(self):
        with jt.log_capture_scope(log_vprefix="setitem_gopt=1000") as rep:
            v = jt.random((4,2,2))
            d = jt.ones((2,2))
            v[1] = d
            del d
            v.sync()
        logs = find_log_with_re(rep, "setitem_inplace happens")
        assert len(logs) == 1

    def test_setitem2(self):
        with jt.log_capture_scope(log_vprefix="setitem_gopt=1000") as rep:
            v = jt.random((4,2,2))
            d = jt.ones((2,2))
            v[1] = d
            v.sync()
        logs = find_log_with_re(rep, "setitem_inplace happens")
        assert len(logs) == 0

    def test_setitem3(self):
        with jt.log_capture_scope(log_vprefix="setitem_gopt=1000") as rep:
            v = jt.random((4,2,2))
            v1 = v[1:3]
            d = jt.ones((2,2))
            v1[1] = d
            v1.sync()
        logs = find_log_with_re(rep, "setitem_inplace happens")
        assert len(logs) == 0
    
    def test_setitem4(self):
        with jt.log_capture_scope(log_vprefix="setitem_gopt=1000") as rep:
            v = jt.random((4,2,2))
            v1 = v[1:3,0]
            d = jt.ones((2,))
            v1[1] = d
            del d
            v1.sync()
        logs = find_log_with_re(rep, "setitem_inplace happens")
        assert len(logs) == 1
    
    def test_setitem5(self):
        with jt.log_capture_scope(log_vprefix="setitem_gopt=1000") as rep:
            v = jt.random((4,2,2))
            v1 = v[1:3,0]
            d = jt.ones((2,2))
            d1 = d[0]
            v1[1] = d1
            v1.sync()
        logs = find_log_with_re(rep, "setitem_inplace happens")
        assert len(logs) == 0
    
    def test_setitem6(self):
        with jt.log_capture_scope(log_vprefix="setitem_gopt=1000") as rep:
            v = jt.random((4,2,2))
            v1 = v[1:3,0]
            d = jt.ones((2,2,2))
            d1 = d[0,0]
            v1[1] = d1
            del d1
            v1.sync()
        logs = find_log_with_re(rep, "setitem_inplace happens")
        assert len(logs) == 1

    def test_getitem1(self):
        with jt.log_capture_scope(log_vprefix="setitem_gopt=1000") as rep:
            v = jt.random((4,3))
            v_res = v[2,:]
            v_res.data[1] = 1
        logs = find_log_with_re(rep, "getitem_inplace happens")
        assert len(logs) == 1

    def test_getitem2(self):
        with jt.log_capture_scope(log_vprefix="setitem_gopt=1000") as rep:
            v1 = jt.array([1,2,3,4])
            v1_res = v1[None]
            v1_res.data[0,2] = -1
        logs = find_log_with_re(rep, "getitem_inplace happens")
        assert len(logs) == 1

    def test_getitem3(self):
        with jt.log_capture_scope(log_vprefix="setitem_gopt=1000") as rep:
            arr3 = jt.array([1,2,3,4])
            arr3_res = arr3[3]
            arr3_res.data[0] = -1
        logs = find_log_with_re(rep, "getitem_inplace happens")
        assert len(logs) == 1

    def test_getitem4(self):
        with jt.log_capture_scope(log_vprefix="setitem_gopt=1000") as rep:
            arr4 = jt.random((4,2,3,3))
            arr4_res = arr4[...,:,:]
            arr4_res.data[0,0,1,1] = 1
        logs = find_log_with_re(rep, "getitem_inplace happens")
        assert len(logs) == 1

    def test_getitem5(self):
        with jt.log_capture_scope(log_vprefix="setitem_gopt=1000") as rep:
            arr5 = jt.random((4,2,3,3))
            arr5_res = arr5[1:3,:,:,:]
            arr5_res.data[1,0,1,1] = 1
        logs = find_log_with_re(rep, "getitem_inplace happens")
        assert len(logs) == 1

    def test_getitem6(self):
        with jt.log_capture_scope(log_vprefix="setitem_gopt=1000") as rep:
            arr6 = jt.random((4,2,3,3))
            arr6_res = arr6[1]
            arr6_res.data[0,1,1] = 1
        logs = find_log_with_re(rep, "getitem_inplace happens")
        assert len(logs) == 1
    
    def test_getitem7(self):
        with jt.log_capture_scope(log_vprefix="setitem_gopt=1000") as rep:
            arr2 = jt.array([1,2,3,4])
            arr2_res = arr2[...]
            arr2_res.data[2] = -1
        logs = find_log_with_re(rep, "getitem_inplace happens")
        assert len(logs) == 1

    def test_getitem8(self):
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
        
if __name__ == "__main__":
    unittest.main()