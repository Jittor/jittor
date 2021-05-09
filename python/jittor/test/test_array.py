# ***************************************************************
# Copyright (c) 2021 Jittor. All Rights Reserved. 
# Maintainers: Dun Liang <randonlang@gmail.com>. 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
import numpy as np
from jittor import compile_extern
from jittor.test.test_core import expect_error

class TestArray(unittest.TestCase):
    def test_data(self):
        a = jt.array([1,2,3])
        assert (a.data == [1,2,3]).all()
        d = a.data
        a.data[1] = -2
        assert (a.data == [1,-2,3]).all()
        assert (a.fetch_sync()==[1,-2,3]).all()
        li = jt.liveness_info()
        del a
        assert li == jt.liveness_info()
        del d
        assert li != jt.liveness_info()

    def test_set_data(self):
        a = jt.array([1,2,3])
        assert (a.fetch_sync()==[1,2,3]).all()
        a.data = [4,5,6]
        assert (a.fetch_sync()==[4,5,6]).all()
        a.data = jt.array([7,8,9])
        assert (a.fetch_sync()==[7,8,9]).all()

    @unittest.skipIf(not jt.has_cuda, "Cuda not found")
    @jt.flag_scope(use_cuda=1)
    def test_memcopy_overlap(self):
        import time
        from jittor.models import resnet

        im=np.random.rand(100,3,224,224).astype(np.float32)
        net = resnet.Resnet34()
        net.eval()
        # warm up
        x = jt.array(im).stop_grad()

        for i in range(10):
            a = net(x)
            a.sync()
        jt.sync(device_sync=True)

        # pure compute
        time_start=time.time()
        x = jt.array(im).stop_grad()
        for i in range(10):
            a = net(x)
            a.sync()
        jt.sync(device_sync=True)
        t1 = time.time() - time_start

        # warm up
        for i in range(3):
            x = jt.array(im)
            b = net(x)
            b.fetch(lambda b: None)
            b.sync()
        jt.sync(device_sync=True)

        # overlap
        time_start=time.time()
        results = []
        for i in range(10):
            x = jt.array(im)
            b = net(x)
            b.fetch(lambda b: results.append(b))
            b.sync()
            # del c
        jt.sync(device_sync=True)
        t2 = time.time() - time_start

        assert t2-t1 < 0.010, (t2, t1, t2-t1)
        assert np.allclose(a.data, b.data)
        assert len(results) == 10
        for v in results:
            assert np.allclose(a.data, v), (v.shape, a.data.shape)
        jt.LOG.v(f"pure compute: {t1}, overlap: {t2}")

    def test_segfault(self):
        a = jt.array([1.0,2.0,3.0])
        b = (jt.maximum(a, 0)).sum() * 2.0
        da = jt.grad(b, a)
        jt.sync_all()
        assert (a.data==[1,2,3]).all()
        assert (da.data==[2,2,2]).all()

    def test_segfault2(self):
        assert (jt.array([1,2,3]).reshape((1,3)).data==[1,2,3]).all()
        if jt.has_cuda:
            with jt.flag_scope(use_cuda=1):
                assert (jt.array([1,2,3]).reshape((1,3)).data==[1,2,3]).all()
    
    @unittest.skipIf(not jt.has_cuda, "Cuda not found")
    def test_array_dual(self):
        with jt.flag_scope(use_cuda=1):
            a = jt.array(np.float32([1,2,3]))
            assert (a.data==[1,2,3]).all()
        
    @unittest.skipIf(not jt.has_cuda, "Cuda not found")
    def test_array_migrate(self):
        with jt.flag_scope(use_cuda=1):
            a = jt.array(np.float32([1,2,3]))
            b = jt.code(a.shape, a.dtype, [a], cpu_src="""
                for (int i=0; i<in0_shape0; i++)
                    @out(i) = @in0(i)*@in0(i)*2;
            """)
            assert (b.data==[2,8,18]).all()
        
    def test_not_c_style(self):
        a = np.array([1,2,3])
        b = a[::-1]
        x = jt.array(b)
        x = x + b
        assert (x.data == [6,4,2]).all()

    def test_scalar(self):
        assert jt.array(1).data == 1
        assert jt.array(np.float64(1)).data == 1
        assert jt.array(np.float32(1)).data == 1
        assert jt.array(np.int32(1)).data == 1
        assert jt.array(np.int64(1)).data == 1

    def test_array_dtype(self):
        a = jt.array([1,2,3], dtype=jt.NanoString("float32"))
        a = jt.array([1,2,3], dtype=jt.float32)

    def test_var(self):
        a = jt.Var([1,2,3])
        b = jt.Var([1,2,3], "float32")
        assert a.dtype == "int32"
        assert b.dtype == "float32"
        assert (a.numpy() == [1,2,3]).all()
        assert (b.numpy() == [1,2,3]).all()

    def test_np_array(self):
        a = jt.Var([1,2,3])
        b = np.array(a)
        assert (b==[1,2,3]).all()

    def test_pickle(self):
        import pickle
        a = jt.Var([1,2,3,4])
        s = pickle.dumps(a, pickle.HIGHEST_PROTOCOL)
        b = pickle.loads(s)
        assert isinstance(b, jt.Var)
        assert (b.data == [1,2,3,4]).all()

    def test_tuple_array(self):
        a = jt.array((4,5))
        expect_error(lambda : jt.array({}))
        expect_error(lambda : jt.array("asdasd"))
        expect_error(lambda : jt.array(jt))

    def test_64_bit(self):
        a = np.random.rand(10)
        b = jt.array(a)
        assert b.dtype == "float32"

        with jt.flag_scope(auto_convert_64_to_32=0):
            a = np.random.rand(10)
            b = jt.array(a)
            assert b.dtype == "float64"

        a = np.random.rand(10)
        b = jt.array64(a)
        assert b.dtype == "float64"


if __name__ == "__main__":
    unittest.main()