
from _helpers import capability as _test_capability
# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved. 
# Maintainers: 
#     Guowei Yang <471184555@qq.com>
#     Dun Liang <randonlang@gmail.com>. 
# 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
from jittor import Function
import jittor as jt
import numpy
import ctypes
import sys
import pytest

from _helpers.state_leaks import assert_rss_growth_bounded

try:
    import cupy
except:
    pass

class TestCodeOp(unittest.TestCase):
    def test_func(self):
        class Func(Function):
            def forward_code(self, np, data):
                a = data["inputs"][0]
                b = data["outputs"][0]
                if (jt.introspection.policy.runtime.use_cuda==0):
                    assert isinstance(a,numpy.ndarray)
                else:
                    assert isinstance(a,cupy.ndarray)
                np.add(a,a,out=b)

            def backward_code(self, np, data):
                a, dout = data["inputs"]
                out = data["outputs"][0]
                np.copyto(out, dout*2.0)

            def execute(self, a):
                self.save_vars = a
                return jt.numpy_code(
                    a.shape,
                    a.dtype,
                    [a],
                    self.forward_code,
                )

            def grad(self, grad_a):
                a = self.save_vars
                return jt.numpy_code(
                    a.shape,
                    a.dtype,
                    [a, grad_a],
                    self.backward_code,
                )

        def check():
            a = jt.random((5,1))
            func = Func()
            b = func(a)
            assert numpy.allclose(b.data,(a+a).data)
            da = jt.grad(b,a)
            one=numpy.ones(a.shape)
            assert numpy.allclose(da.data,one*2.0)

        if _test_capability.check_accelerator('cuda', backend=jt).enabled:
            with jt.flag_scope(use_cuda=1):
                check()
        check()

    def test(self):
        def forward_code(np, data):
            a = data["inputs"][0]
            b = data["outputs"][0]
            if (jt.introspection.policy.runtime.use_cuda==0):
                assert isinstance(a,numpy.ndarray)
            else:
                assert isinstance(a,cupy.ndarray)
            np.add(a,a,out=b)

        def backward_code(np, data):
            dout = data["dout"]
            out = data["outputs"][0]
            np.copyto(out, dout*2.0)

        def check():
            a = jt.random((5,1))
            b = jt.numpy_code(
                a.shape,
                a.dtype,
                [a],
                forward_code,
                [backward_code],
            )
            assert numpy.allclose(b.data,(a+a).data)
            da = jt.grad(b,a)
            one=numpy.ones(a.shape)
            assert numpy.allclose(da.data,one*2.0)

        if _test_capability.check_accelerator('cuda', backend=jt).enabled:
            with jt.flag_scope(use_cuda=1):
                check()
        check()

    def test_multi_input(self):
        def forward_code(np, data):
            a,b = data["inputs"]
            c,d = data["outputs"]
            np.add(a,b,out=c)
            np.subtract(a,b,out=d)

        def backward_code1(np, data):
            dout = data["dout"]
            out = data["outputs"][0]
            np.copyto(out, dout)

        def backward_code2(np, data):
            dout = data["dout"]
            out_index = data["out_index"]
            out = data["outputs"][0]
            if out_index==0:
                np.copyto(out, dout)
            else:
                np.negative(dout, out)

        def check():
            a = jt.random((5,1))
            b = jt.random((5,1))
            c, d = jt.numpy_code(
                [a.shape, a.shape],
                [a.dtype, a.dtype],
                [a, b],
                forward_code,
                [backward_code1,backward_code2],
            )
            assert numpy.allclose(c.data,(a+b).data)
            assert numpy.allclose(d.data,(a-b).data)
            dca, dcb = jt.grad(c,[a,b])
            dda, ddb = jt.grad(d,[a,b])
            one=numpy.ones(a.shape)
            mone=one*-1.0
            assert numpy.allclose(dca.data,one)
            assert numpy.allclose(dcb.data,one)
            assert numpy.allclose(dda.data,one)
            assert numpy.allclose(ddb.data,mone)
        
        if _test_capability.check_accelerator('cuda', backend=jt).enabled:
            with jt.flag_scope(use_cuda=1):
                check()
        check()

    def test_backward_dout_is_device_resident(self):
        # The gradient seed handed to a numpy_code backward is a stride-0
        # broadcast of a number, so make_numpy_code's adapt_storage_input
        # replaces it with a contiguous copy before the op is built. The op used
        # to keep the pre-replacement Var and read its mem_ptr at run time --
        # null by then, which makes PyArray_New allocate a *host* buffer that
        # numpy2cupy hands to CuPy as a device address. Asserting the value
        # alone is not enough: cupy.copyto off a host pointer returns a wrong
        # gradient without raising, and only an arithmetic kernel takes the
        # context down.
        seen = {}

        def forward_code(np, data):
            np.add(data["inputs"][0], data["inputs"][0], out=data["outputs"][0])

        def backward_code(np, data):
            dout = data["dout"]
            seen["dout"] = int(dout.__array_interface__["data"][0]) \
                if isinstance(dout, numpy.ndarray) else int(dout.data.ptr)
            np.copyto(data["outputs"][0], dout * 2.0)

        def check(on_cuda):
            seen.clear()
            a = jt.random((5, 1))
            b = jt.numpy_code(a.shape, a.dtype, [a], forward_code,
                              [backward_code])
            da = jt.grad(b, a)
            assert numpy.allclose(da.data, numpy.ones(a.shape) * 2.0)
            assert "dout" in seen
            if on_cuda:
                # 0 is cudaMemoryTypeUnregistered: plain host memory.
                kind = cupy.cuda.runtime.pointerGetAttributes(seen["dout"]).type
                assert kind != 0, \
                    "numpy_code backward received a host pointer for dout"

        if _test_capability.check_accelerator('cuda', backend=jt).enabled:
            with jt.flag_scope(use_cuda=1):
                check(True)
        check(False)

    @pytest.mark.slow
    def test_memory_leak(self):
        def forward_code(np, data):
            a,b = data["inputs"]
            c,d = data["outputs"]
            np.add(a,b,out=c)
            np.subtract(a,b,out=d)

        def backward_code1(np, data):
            dout = data["dout"]
            out = data["outputs"][0]
            np.copyto(out, dout)

        def backward_code2(np, data):
            dout = data["dout"]
            out_index = data["out_index"]
            out = data["outputs"][0]
            if out_index==0:
                np.copyto(out, dout)
            else:
                np.negative(dout, out)

        def exercise():
            a = jt.random((4096, 1))
            b = jt.random((4096, 1))
            c, d = jt.numpy_code(
                [a.shape, a.shape],
                [a.dtype, a.dtype],
                [a, b],
                forward_code,
                [backward_code1,backward_code2],
            )
            assert numpy.allclose(c.data,(a+b).data)
            assert numpy.allclose(d.data,(a-b).data)
            dca, dcb = jt.grad(c,[a,b])
            dda, ddb = jt.grad(d,[a,b])
            one=numpy.ones(a.shape)
            mone=one*-1.0
            assert numpy.allclose(dca.data,one)
            assert numpy.allclose(dcb.data,one)
            assert numpy.allclose(dda.data,one)
            assert numpy.allclose(ddb.data,mone)

        assert_rss_growth_bounded(
            exercise, iterations=128, max_growth_bytes=8 << 20,
            cleanup=jt.clean)

if __name__ == "__main__":
    unittest.main()
